#!/usr/bin/env python
# coding=utf-8
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from medfound.data.io import read_data
from medfound.models.load_model import ModelArguments
from medfound.data.prompt import SftPrompter
import logging
import fire
from transformers import (
    Trainer, HfArgumentParser, TrainingArguments,
)
from medfound.utils import T5_MODEL_TYPES
from medfound.trainer import TrainerFramework

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="bitsandbytes.cuda_setup")
logger = logging.getLogger(__name__)


@dataclass
class GenerateArguments:
    policy: str = field(
        default="default",
        metadata={"help": "generation policy"},
    )

    # base arguments
    temperature: float = field(
        default=0.7,
        metadata={"help": "temperature. 𝑞𝑖=exp(𝑧𝑖/𝑇)/∑𝑗exp(𝑧𝑗/𝑇). "},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "top_p"},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "num_return_sequences"}
    )

    # policy
    num_beams: int = field(
        default=None,
        metadata={"help": "num_beams"},
    )
    do_sample: bool = field(
        default=None,
        metadata={"help": "do_sample"},
    )
    penalty_alpha: float = field(
        default=None,
        metadata={"help": "penalty_alpha"},
    )
    top_k: int = field(
        default=None,
        metadata={"help": "top_k"},
    )
    num_beam_groups: int = field(
        default=None,
        metadata={"help": "num_beam_groups"},
    )

    policy_args: dict = field(
        default=None,
        metadata={"help": "policy_args"},
    )

    def __post_init__(self):
        # Utilities for Generation
        # https://huggingface.co/docs/transformers/internal/generation_utils

        if self.policy == 'default':
            policy = dict()
        elif self.policy == 'greedy':  # Greedy search
            policy = dict(num_beams=1, do_sample=False)
        elif self.policy == 'sample':  # Multinomial sampling
            policy = dict(num_beams=1, do_sample=True)
        elif self.policy == 'contrastive':  # Contrastive search
            policy = dict(penalty_alpha=0.6, top_k=4)
        elif self.policy == 'beam_search':  # Beam search
            policy = dict(num_beams=5, do_sample=False)
        elif self.policy == 'beam_sample':  # Beam-search multinomial sampling
            policy = dict(num_beams=5, do_sample=True)
        elif self.policy == 'group_beam_search':
            # Diverse beam search decoding
            policy = dict(num_beams=5, num_beam_groups=5)
        else:
            raise ValueError("check generation_strategies")

        # generate args order
        # 1. policy default args
        # 2. manual args overrides default args
        policy = self._update_args(policy)
        policy['temperature'] = self.temperature
        policy['top_p'] = self.top_p
        policy['num_return_sequences'] = self.num_return_sequences
        self.policy_args = policy

    def _update_args(self, policy):
        for k in policy.keys():
            if self.__getattribute__(k) is not None:
                policy[k] = self.__getattribute__(k)
        return policy


def make_supervised_data_module(tokenizer, data_args,
                                model_type, lm_type=None):
    """
    Generate supervised data module for training.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        data_args: The arguments for data processing.
        model_type (str): Type of the model.
        lm_type (str, optional): Type of language model. Defaults to None.

    Returns:
        Tuple[SupervisedDataset, SupervisedDataset,
        DataCollatorForSupervisedDataset]
    """
    from medfound.data.sft_dataset import SupervisedDataset
    from medfound.data.sft_dataset import \
        DataCollatorForSupervisedDataset

    if lm_type is None:
        lm_type = 'dec'
        if model_type in T5_MODEL_TYPES:
            lm_type = 'encdec'

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:
            logger.warning(
                "The tokenizer picked seems to have a very large "
                f"`model_max_length` ({tokenizer.model_max_length}). "
                "Picking 4096 instead. You can change that default value by "
                "passing --block_size xxx."
            )
            block_size = 4096
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger "
                "than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using "
                f"block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_dataset = None
    if data_args.data_path:
        train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          lm_type=lm_type,
                                          block_size=block_size,
                                          prompt_name=data_args.prompt_name,
                                          add_qid=data_args.add_qid,
                                          cache_dir=data_args.cache_dir,
                                          num_proc=data_args.num_proc)

    eval_dataset = None
    if data_args.data_path_test:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer,
                                         data_path=data_args.data_path_test,
                                         lm_type=lm_type,
                                         block_size=block_size,
                                         prompt_name=data_args.prompt_name,
                                         add_qid=data_args.add_qid,
                                         cache_dir=data_args.cache_dir,
                                         num_proc=data_args.num_proc)
    else:
        eval_dataset = train_dataset

    if model_type in ['glm', 'chatglm', 'cpmant']:
        add_causal_mask = True
    else:
        add_causal_mask = False
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        add_causal_mask=add_causal_mask)
    return train_dataset, eval_dataset, data_collator


compute_metrics = None
preprocess_logits_for_metrics = None


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss.

        Args:
            model: The model to compute the loss for.
            inputs: The inputs to the model.
            return_outputs (bool): Whether to return the outputs along with
            the loss. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ModelOutput]]: The
            computed loss value.
        """
        raise super(CustomTrainer, self).compute_loss(model,
                                                      inputs, return_outputs)


class CLMRunner(object):
    def __init__(self):
        from medfound.data.sft_dataset import DataArguments
        self.trainer = TrainerFramework(
            task='lm', trainer_class=Trainer,
            make_supervised_data_module=make_supervised_data_module,
            model_args_class=ModelArguments,
            data_args_class=DataArguments,)

    def train(self, config=None, **kargs):
        self.trainer.train(config=config, **kargs)

    def export(self, output_dir, config=None,
               resume_from_checkpoint=None, **kargs):
        """Export a model

        Args:
            output_dir (str or Path): Output directory.
            config (optional): Configuation. Defaults to None.
            resume_from_checkpoint (str, optional): Checkpoint path.
            Defaults to None.
        """
        self.trainer.export(output_dir, config=config,
                            resume_from_checkpoint=resume_from_checkpoint,
                            **kargs)

    def generate(self, data_path_test, model_name_or_path,
                 max_length=200, config=None,
                 tokenizer_mode='slow',
                 seed=42, trust_remote_code=True, dtype='float16',
                 head=None, prompt_name=None, resume=False,
                 col_id='index',
                 split_dump_size=64, n_jobs=1, i_job=-1,
                 quantization=None,
                 tensor_parallel_size=None,
                 verbose=True, **kargs):
        """Generate outputs using vllm.

        Args:
            data_path_test (str): The path to the test data file.
            model_name_or_path (str): The name or path of model.
            max_length (int, optional): : The maximum length of generated
            outputs. Defaults to 200.
            config (str or dict, optional): Model configuration.
            Defaults to None.
            tokenizer_mode (str, optional): The mode of tokenizer.
            Defaults to 'slow'.
            seed (int, optional): The seed. Defaults to 42.
            trust_remote_code (bool, optional): Whether to trust remote code.
            Defaults to True.
            dtype (str, optional): Data type. Defaults to 'float16'.
            head (str, optional): DataFrame head. Defaults to None.
            prompt_name (str, optional): Name of prompt template used for
            generation. Defaults to None.
            resume (bool, optional): Whether to resume. Defaults to False.
            col_id (str, optional): Column id of test data.
            Defaults to 'index'.
            split_dump_size (int, optional): The size of data dumps.
            Defaults to 64.
            n_jobs (int, optional): World size. Defaults to 1.
            i_job (int, optional): Local rank. Defaults to -1.
            quantization (str, optional): The name of quantization algorithm.
            Defaults to None.
            tensor_parallel_size (int, optional): The number of tensor
            parallel. Defaults to None.
            verbose (bool, optional): Whether to print details logs.
            Defaults to True.

        Raises:
            e: _description_
        """
        from vllm import LLM, SamplingParams

        # parse generate args
        parser = HfArgumentParser((GenerateArguments, TrainingArguments))
        if config:
            kargs_config = json.loads(Path(config).read_text())
            kargs_config.update(kargs)
            kargs = kargs_config
            job_name = kargs.pop('job_name', None)
            if job_name is not None:
                kargs['output_dir'] = f'{kargs["output_dir"]}/{job_name}'
        generate_args, training_args = \
            parser.parse_dict(kargs, allow_extra_keys=True)
        policy = generate_args.policy
        policy_args = generate_args.policy_args
        rename_map = dict(
            num_return_sequences='n',
            num_beams='best_of',
            penalty_alpha='presence_penalty',
        )
        policy_args = {rename_map.get(k, k): v for k, v in policy_args.items()}
        policy_args.pop('do_sample', None)
        policy_args.pop('num_beam_groups', None)
        try:
            if (Path(model_name_or_path)/'config.json').exists():
                generation_config = json.loads((
                    Path(model_name_or_path)/'config.json').read_text())
                policy_args['stop_token_ids'] = \
                    [generation_config['eos_token_id']]
        except Exception as e:
            logger.warning(f'Error in {model_name_or_path}/config.json: {e}')
            raise e
        sampling_params = SamplingParams(max_tokens=max_length, **policy_args)

        # load data
        paths = data_path_test.split(',')
        dfs_temp = []
        for path in paths:
            df = read_data(path)
            df['data_source'] = Path(path).name
            dfs_temp.append(df)
        df = pd.concat(dfs_temp)

        # data preparation
        prompter = SftPrompter(template_name=prompt_name)
        temp = df['text'].apply(
            lambda x: prompter.generate_prompt_response(x, with_last=False))
        df['prompt'] = [x[0] for x in temp]
        df['reference'] = [x[1] for x in temp]

        # split data in multi-GPU processing
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        if n_jobs > 1:
            world_size = n_jobs
            local_rank = i_job
        if world_size > 1:
            df = df.iloc[local_rank::world_size]

        # output dir
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if local_rank != -1:
            filename = f'generate.{local_rank}.{world_size}.jsonl'
        else:
            filename = 'generate.0.1.jsonl'
        output_path = output_dir / filename

        if col_id not in df.columns:
            warnings.warn(f'Column {col_id} not found in {data_path_test}, '
                          'use index instead')
            df[col_id] = df.index
        if resume and output_path.exists():
            df_resume = pd.read_json(output_path, lines=True)
            if col_id not in df_resume.columns:
                logger.warning(f'Column {col_id} not found in {output_path}, '
                               'ignore resume')
            else:
                df = df[~df[col_id].isin(df_resume[col_id])]
        else:
            output_path.write_text('')

        if head:
            df = df.head(head)
        df = df.reset_index(drop=True)  # reset the index

        # model args
        llm_kargs = {}
        llm_kargs['tokenizer_mode'] = tokenizer_mode
        llm_kargs['max_model_len'] = max_length
        if tensor_parallel_size is not None:
            llm_kargs['tensor_parallel_size'] = tensor_parallel_size
        if quantization is not None:
            llm_kargs['quantization'] = quantization

        # load the model
        llm = LLM(model=model_name_or_path, seed=seed,
                  trust_remote_code=trust_remote_code, dtype=dtype,
                  **llm_kargs)
        # chunk the generation
        if split_dump_size > 0:
            n_splits = min((len(df)-1) // split_dump_size + 1, len(df))
            dfs = np.array_split(df, n_splits)
        else:
            dfs = [df]

        pbar = tqdm(total=len(df), disable=not verbose)
        for i, df_sub in enumerate(dfs):
            try:
                outputs = llm.generate(df_sub['prompt'].tolist(),
                                       sampling_params, use_tqdm=False)
            except Exception as e:
                logger.error(f'Error in {i}th batch: {e}')
                continue
            outputs = [o for o in outputs
                       if len(o.outputs) == policy_args['n']]
            for i_seq in range(policy_args['n']):
                df_output_temp = pd.DataFrame({
                    'id': [o.request_id for o in outputs],
                    'response': [o.outputs[i_seq].text for o in outputs],
                    'finish_reason': [o.outputs[i_seq].finish_reason
                                      for o in outputs],
                    'prompt_token_len': [len(o.prompt_token_ids)
                                         for o in outputs],
                    'response_token_len': [len(o.outputs[i_seq].token_ids)
                                           for o in outputs],
                })
                df_output_temp = df_output_temp.drop_duplicates(subset=['id'])
                df_output_temp['id'] = df_output_temp['id'].\
                    astype(df_sub.index.dtype)
                df_output_temp = df_output_temp.set_index('id')

                df_sub['finish_reason'] = df_output_temp['finish_reason']
                df_sub['response'] = df_output_temp['response']
                df_sub['response'] = df_sub['response'].fillna('')
                df_sub['response_token_len'] = \
                    df_output_temp['response_token_len']
                df_sub['prompt_token_len'] = df_output_temp['prompt_token_len']

                # generation args
                df_sub['i'] = i_seq
                df_sub['model_name'] = model_name_or_path
                df_sub['policy_name'] = policy
                for k, v in policy_args.items():
                    if isinstance(v, list):
                        continue
                    df_sub[k] = v

                # convert DataFrame to JSON string
                json_str = df_sub.to_json(orient='records', lines=True,
                                          force_ascii=False)

                # append the JSON string to the file
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(json_str)
                    f.write('\n')

            pbar.update(len(df_sub))


if __name__ == '__main__':
    fire.Fire(CLMRunner)
