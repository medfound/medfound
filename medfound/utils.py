import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path
import datasets
import pandas as pd
import torch
import transformers
from transformers import Trainer
import re
from medfound.models.load_model import ModelArguments, load_model
import numpy as np
from functools import lru_cache

T5_MODEL_TYPES = ['t5', 'mt5']
logger = logging.getLogger(__name__)

"""
Helpers to support streaming generate output. Borrowed from
https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/callbacks.py
"""


lora_full_config = {
    'bloom': ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    'llama': ['q_proj', 'k_proj', 'v_proj',
              'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
}


def lora_load_and_merge(task, peft_model_path, base_model_path=None,
                        cache_dir=None, torch_dtype=None,
                        safe_merge=False, **model_kargs):
    """Load and Merge LoRA Model

    Args:
        task (str): Task name.
        peft_model_path (str or Path): Path of peft model.
        base_model_path (str or Path, optional): Path of peft model.
        Defaults to None.
        cache_dir (str or Path, optional): The directory to cache.
        Defaults to None.
        torch_dtype (optional): Data type for PyTorch tensors.
        Defaults to None.
        safe_merge (bool, optional): Whether to merge safely.
        Defaults to False.

    Raises:
        ValueError: If an unsupported task is provided.

    Returns:
        torch.nn.Module: Merged model of LoRA. Transform all to Hugging Face
        Transformers model for consistency and ease of use.
    """
    import torch
    from transformers import AutoModelForCausalLM
    import peft
    from peft import PeftModel, PeftConfig
    # version requirement of peft
    assert '.'.join(peft.__version__.split('.')[:2]) > '0.3'

    # Load peft config
    peft_config = PeftConfig.from_pretrained(peft_model_path)

    if base_model_path is None:
        base_model_path = peft_config.base_model_name_or_path

    torch_dtype = (
            torch_dtype
            if torch_dtype in ["auto", None]
            else getattr(torch, torch_dtype)
        )

    # load base model
    kargs = model_kargs.copy()
    if task == 'lm':
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch_dtype, cache_dir=cache_dir,
            **kargs)
    else:
        raise ValueError(f"Unknown task {task}")
    model = PeftModel.from_pretrained(model_base, peft_model_path, **kargs)
    model_base_merged = model.merge_and_unload(safe_merge=safe_merge)
    return model_base_merged


def lora_merge_export(peft_model_path, output_dir,
                      base_model_path=None, cache_dir=None,
                      task='lm', device='cpu', safe_merge=False, **kargs):
    """Merge and export LoRA model.

    Args:
        peft_model_path (str or Path): Path of peft model.
        output_dir (str or Path): Output directory.
        base_model_path (str or Path, optional): Path of peft model.
        Defaults to None.
        cache_dir (str or Path, optional): The directory to cache.
        Defaults to None.
        task (str, optional): Task name. Defaults to 'lm'.
        device (str, optional): Device. Defaults to 'cpu'.
        safe_merge (bool, optional): Whether to merge safely.
        Defaults to False.
    """
    # load and merge LoRA model
    model_base_merged = lora_load_and_merge(
        task, peft_model_path, base_model_path=base_model_path,
        cache_dir=cache_dir, safe_merge=safe_merge, **kargs)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_base_merged.save_pretrained(output_dir)


def export(output_dir, task, config=None,
           resume_from_checkpoint=None, **kargs):
    """Export a model

    Args:
        output_dir (str or Path): Output directory.
        task (str): Task name.
        config (optional): Configuation. Defaults to None.
        resume_from_checkpoint (str, optional): Checkpoint path.
        Defaults to None.

    Raises:
        ValueError: Checkpoint path does not exist.
    """
    from transformers import HfArgumentParser
    parser = HfArgumentParser((ModelArguments,))
    if resume_from_checkpoint is not None:
        if not Path(resume_from_checkpoint).exists():
            raise ValueError(
                f"Can't find checkpoint {resume_from_checkpoint}")

    if config:
        (model_args,) = parser.parse_json_file(
            json_file=os.path.abspath(config))
    else:
        (model_args,) = parser.parse_dict(kargs)

    # Load pretrained model an∑d tokenizer
    model, tokenizer = load_model(task, model_args, model_args.local_dir)

    # export
    trainer = Trainer(model=model, tokenizer=tokenizer)
    if resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(resume_from_checkpoint)

    export_trainer_model(trainer, output_dir, use_lora=model_args.lora,
                         max_shard_size="4GB")


def _zero3_consolidated_16bit_state_dict(engine, trainable_only=False):
    """
    Get a full non-partitioned state_dict with fp16 weights on cpu.
    Important: this function must be called on all ranks and not just rank 0.
    This is similar to nn.Module.state_dict (modelled after
    _save_to_state_dict), but:
    1. consolidates the weights from different partitions on gpu0
    2. works on one layer at a time to require as little gpu0 memory as
    possible, by moving the already consolidated weights to cpu
    3. takes care to keep the shared params shared when gradually copying the
    params to cpu
    Returns:
        a consolidated fp16 ``state_dict`` on cpu on rank 0, ``None`` on other
        ranks

    trainable_only: bool
        - for lora, export only trainable parameters
        - save memory
    """
    if not engine.zero_optimization_partition_weights():
        raise ValueError("this function requires ZeRO-3 mode")

    from collections import OrderedDict
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.utils import see_memory_usage

    state_dict = OrderedDict() if dist.get_rank() == 0 else None
    shared_params = {}

    def get_layer_state_dict(module, prefix=""):
        # gather one layer at a time to be memory-efficient
        # must use modifier_rank=0 to release GPU memory after each layer
        # gathered see_memory_usage("before GatheredParameters", force=True)

        with deepspeed.zero.GatheredParameters(
             list(module.parameters(recurse=False)), modifier_rank=0):
            if dist.get_rank() == 0:
                # handle params
                for name, param in module.named_parameters(recurse=False):
                    if param is None:
                        continue
                    if trainable_only and not param.requires_grad:
                        continue
                    key = prefix + name
                    # can't rely on param.data_ptr() as it will be reused as
                    # weights gets gathered and reduced, but param.ds_id is
                    # unique across all zero weights (and shared params will
                    # have the same param.ds_id)
                    if param.ds_id in shared_params:
                        # shared weights
                        state_dict[key] = \
                            state_dict[shared_params[param.ds_id]]
                    else:
                        state_dict[key] = param.detach().cpu()
                        shared_params[param.ds_id] = key

                # now buffers - not sure if need to take care of potentially
                # shared weights here
                for name, buf in module.named_buffers(recurse=False):
                    if (buf is not None and
                            name not in module._non_persistent_buffers_set):
                        state_dict[prefix + name] = buf.detach().cpu()
        # see_memory_usage("after GatheredParameters", force=True)

        for name, child in module.named_children():
            if child is not None:
                get_layer_state_dict(child, prefix + name + ".")

    # Prepare for checkpoint save by ensuring all parameters are partitioned
    if engine._optimizer_has_ckpt_event_prologue():
        engine.optimizer.checkpoint_event_prologue()

    see_memory_usage("before get_layer_state_dict", force=False)
    get_layer_state_dict(engine.module, prefix="")
    see_memory_usage("after get_layer_state_dict", force=False)

    if engine._optimizer_has_ckpt_event_epilogue():
        engine.optimizer.checkpoint_event_epilogue()

    return state_dict


def export_trainer_model(trainer, output_dir, use_lora=False,
                         load_in_kbit=False, max_shard_size="4GB"):
    """
    Export a trained model. Supports options for LoRA and model quantization.

    Args:
        trainer: The training object.
        output_dir (str): The output directory path.
        use_lora (bool, optional): Whether to usr LoRA. Defaults to False.
        load_in_kbit (bool, optional): Whether to use model quantization.
        Defaults to False.
        max_shard_size (str, optional): The maximum shard size.
        Defaults to "4GB".
    """
    trainer.tokenizer.save_pretrained(output_dir)
    is_zero3 = trainer.is_deepspeed_enabled and \
        trainer.accelerator.deepspeed_config["zero_optimization"]["stage"] == 3

    if is_zero3:
        warnings.warn("Zero3 model cannot be merged in training process, "
                      "please merge it manually.")

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        state_dict = _zero3_consolidated_16bit_state_dict(trainer.deepspeed,
                                                          trainable_only=True)
        unwrapped_model.save_pretrained(
            str(Path(output_dir) / 'adapter_backup'),
            state_dict=state_dict,
            max_shard_size=max_shard_size,
        )
        return

    if use_lora:
        model = trainer.model
        model.save_pretrained(str(Path(output_dir) / 'adapter_backup'),
                              max_shard_size=max_shard_size,
                              save_embedding_layers=True,
                              safe_serialization=False)

        # merge and export adapter
        if load_in_kbit:
            warnings.warn(
                "Quantized model cannot be merged, please merge it manually.")
            return

        else:
            if hasattr(trainer.model, 'merge_and_unload'):
                trainer.model = trainer.model.merge_and_unload()
            else:
                logger.warning("Model does not have merge_and_unload function,"
                               " please check.")

    trainer.model.save_pretrained(output_dir, max_shard_size=max_shard_size)

    if use_lora:
        model = trainer.model
        model.save_pretrained(str(Path(output_dir) / 'adapter_backup'),
                              max_shard_size=max_shard_size,
                              save_embedding_layers=True,
                              safe_serialization=False)

        # merge and export adapter
        if load_in_kbit:
            warnings.warn("Quantized model cannot be merged, "
                          "please merge it manually.")
            return

        else:
            if hasattr(trainer.model, 'merge_and_unload'):
                trainer.model = trainer.model.merge_and_unload()
            else:
                logger.warning("Model does not have merge_and_unload function,"
                               " please check.")

    trainer.model.save_pretrained(output_dir, max_shard_size=max_shard_size)


@contextlib.contextmanager
def main_process_first(local_rank, world_size, desc="work", logger=None):
    """
    A context manager for torch distributed environment where on needs to do
    something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.

    One such use is for `datasets`'s `map` feature which to be efficient
    should be run once on the main process,
    which upon completion saves a cached version of results and which then
    automatically gets loaded by the
    replicas.

    Args:
        local_rank (int): Identifier for this process's rank within the world
        of parallel processes.
        world_size (int): The total number of parallel processes involved in
        the operation.
        desc (str, optional): A description of the work being performed.
        Defaults to "work".
        logger (logging.Logger, optional): A logging object to which debug
        information can be written. Defaults to None.
    """
    if world_size > 1:
        main_process_desc = "main process"
        is_main_process = local_rank == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                if logger is not None:
                    logger.debug(f"{local_rank}: waiting for the "
                                 f"{main_process_desc} to perform {desc}")
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                if logger is not None:
                    logger.debug(f"{local_rank}: {main_process_desc} "
                                 f"completed {desc}, releasing all replicas")
                torch.distributed.barrier()
    else:
        yield


def setup_logging(logger, log_level=logging.INFO):
    """Setup logging.
    Args:
        logger (logging.Logger): The logger instance for the application.
        log_level (int, optional): The logging level to set for the
        application and libraries. Defaults to logging.INFO.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def dump_generate_result(results, output_path):
    """
    Save the text generation results to a file in a specified format.

    Args:
        results (list of dict): The generated text and corresponding metadata.
        output_path (str or Path): Output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix not in ['.xlsx', '.xls', '.json', '.jsonl', '.csv']:
        output_path = output_path / 'generate.jsonl'
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_data = pd.DataFrame(results)
    if output_path.suffix in ['.xlsx', '.xls']:
        df_data.to_excel(output_path, index=False)
    elif output_path.suffix in ['.json', '.jsonl']:
        json_str = df_data.to_json(
            orient='records', lines=True, force_ascii=False)
        # append JSON string to file
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json_str)
            f.write('\n')
    else:
        df_data.to_csv(output_path, index=False)


@lru_cache(maxsize=10)
def get_code_mapper(ref_path, model_path, device='cpu'):
    """Get ICD code mapper.

    Args:
        ref_path (str): ICD map file path.
        model_path (str): Model path.
        device (str, optional): 'cpu' or 'cuda'. Defaults to 'cpu'.

    Returns:
        mapper: code mapper
    """
    df_mapper = pd.read_csv(ref_path)
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline

    class DenseTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, model_path, batch_size=4):
            super().__init__()
            import transformers
            self.batch_size = batch_size
            self.encoder = transformers.AutoModel.from_pretrained(model_path)
            self.tokenizer = \
                transformers.AutoTokenizer.from_pretrained(model_path)

        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            embeddings = []
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt',
                                        padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.encoder(**inputs)
                embeddings.append(
                    outputs.last_hidden_state[:, 0].cpu().numpy())
            X = np.concatenate(embeddings)
            return X

    class DiagMapper(BaseEstimator, TransformerMixin):
        def __init__(self, pipeline, df_map):
            super().__init__()
            self.df_map = df_map
            self.pipeline = pipeline
            self.id2code = df_map['diag_code']

        def fit(self, X, y=None, **fit_params):
            return self

        def transform(self, X, y=None, **fit_params):
            idx = self.pipeline.predict(X)
            codes = self.id2code[idx].values
            return codes

    import faiss

    class FaissKNeighbors:
        def __init__(self, n_neighbors=1):
            self.index = None
            self.y = None
            self.k = n_neighbors

        def fit(self, X, y):
            self.index = faiss.IndexFlat(X.shape[1], faiss.METRIC_L1)
            self.index.add(X.astype(np.float32))
            if device == 'cuda':
                res = faiss.StandardGpuResources()  # use a single GPU
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            self.y = y.values

        def predict(self, X):
            distances, indices = self.index.search(X.astype(np.float32),
                                                   k=self.k)
            votes = self.y[indices]
            predictions = votes[:, 0]
            return predictions

    pipeline = Pipeline([
        ('dense', DenseTransformer(model_path)),
        ('knn', FaissKNeighbors(n_neighbors=1)),
    ])

    # train pipeline
    pipeline.fit(df_mapper['diag'], df_mapper.index)
    mapper = DiagMapper(pipeline, df_mapper)

    return mapper


diag_patterns = [
    r'The most likely diagnosis is [^,]+, specifically ([^,\.]+).',
    r'The most likely diagnosis is[:]?([^,\.]+)',
    r'The probable diagnosis is([^,\.]+)',
    r'([^,\.]+) is the most likely diagnosis',
    r'Diagnosis is[:]?([^,\.]+)',
    r'Diagnosis shows([^,\.]+)',
    r'Diagnosis may be[:]?([^,\.]+)',
    r'Likely suffering from[:]?([^,\.]+)',
    r'^([^,\.\s]+)'
]

diag_patterns = [re.compile(p, re.DOTALL) for p in diag_patterns]


def extract_response_diag(response):
    """
    Extract predictions for diagnosis from generated response using
    regularization.

    Args:
        response (str): The generated response string.

    Returns:
        Dict: A dict containing predictions for diagnosis.
    """
    pred = None
    for pattern in diag_patterns:
        results = re.findall(pattern, response)
        if len(results) > 0:
            pred = results[0]
            break
    result = {'pred': pred}
    return result
