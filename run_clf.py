#!/usr/bin/env python
# coding=utf-8
import json
import os
import warnings
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from medfound.models.load_model import ModelArguments, load_model
from medfound.utils import dump_generate_result
from medfound.data.clf_dataset import DataArguments
import logging
import fire
from transformers import Trainer
from medfound.trainer import TrainerFramework

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="bitsandbytes.cuda_setup")
logger = logging.getLogger(__name__)


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
        Tuple[SupervisedClassificationDataset, SupervisedClassificationDataset,
        DataCollatorForSupervisedDataset]
    """
    from medfound.data.clf_dataset import SupervisedClassificationDataset
    from medfound.data.clf_dataset import DataCollatorForSupervisedDataset
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
                f"({tokenizer.model_max_length}). "
                f"Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_dataset = None
    if data_args.data_path:
        train_dataset = SupervisedClassificationDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            block_size=block_size,
            add_qid=data_args.add_qid,
            cache_dir=data_args.cache_dir,
            num_proc=data_args.num_proc)
    if data_args.data_path_test:
        eval_dataset = SupervisedClassificationDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path_test,
            block_size=block_size,
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
        tokenizer=tokenizer, add_causal_mask=add_causal_mask)
    return train_dataset, eval_dataset, data_collator


def compute_metrics(eval_preds):
    """Compute evaluation metrics using sklearn.

    Args:
        eval_preds (tuple): A tuple containing logits and labels.

    Returns:
        dict: A dictionary containing computed evaluation metrics.
              - 'acc' (float): Accuracy score.
    """
    logits, labels = eval_preds
    if isinstance(logits, torch.Tensor):
        logits = logits.to('cpu')
        logits = logits.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.to('cpu')
        labels = labels.numpy()

    # sklearn accuracy
    from sklearn.metrics import accuracy_score
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    result = {'acc': acc}
    return result


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
        return super(CustomTrainer, self).compute_loss(model, inputs,
                                                       return_outputs)


class ClassificationRunner(object):
    def __init__(self):

        trainer_kargs = dict(
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.trainer = TrainerFramework(
            task='clf', trainer_kargs=trainer_kargs,
            trainer_class=CustomTrainer,
            make_supervised_data_module=make_supervised_data_module,
            model_args_class=ModelArguments,
            data_args_class=DataArguments)

    def train(self, config=None, **kargs):
        self.trainer.train(config=config, **kargs)

    def inference(self, config=None, seed=42, batch_size=8, num_workers=1,
                  split_dump_size=None, add_qid=True,
                  resume=False, resume_exclude_all=False,
                  **kargs):

        # TrainingArguments used to initialize the distributed environment
        from transformers import HfArgumentParser
        from transformers import TrainingArguments
        parser = HfArgumentParser((ModelArguments, DataArguments,
                                   TrainingArguments))
        if config:
            if ',' in config:
                configs = config.split(',')
            else:
                configs = [config]
            kargs_config = {}
            for config_path in configs:
                # 1. load config first
                kargs_config_temp = json.loads(Path(config_path).read_text())
                kargs_config.update(kargs_config_temp)
            # 2. command line args override
            kargs_config.update(kargs)
            kargs = kargs_config
            job_name = kargs.pop('job_name', None)
            if job_name is not None:
                kargs['output_dir'] = f'{kargs["output_dir"]}/{job_name}'
        kargs['add_qid'] = add_qid
        if kargs.get('lora', True):
            logger.warning('You are using LoRA model, please set '
                           'lora=False when inference.')
        kargs['lora'] = False
        model_args, data_args, training_args = parser.parse_dict(kargs)

        # Set seed before initializing model.
        from transformers import set_seed
        set_seed(seed)

        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if local_rank != -1:
            filename = f'inference.{local_rank}.{world_size}.jsonl'
        else:
            filename = 'inference.jsonl'
        output_path = output_dir / filename

        # Load pretrained model and tokenizer
        # ** task related
        task = 'clf'
        model_args.lora = False
        model, tokenizer = load_model(task=task, model_args=model_args)

        model.eval()
        import medfound
        with medfound.utils.main_process_first(local_rank=local_rank,
                                               world_size=world_size,
                                               desc="work", logger=None):
            if data_args.data_path_test is not None:
                data_args.data_path = data_args.data_path_test
                data_args.data_path_test = None
            dataset, _, data_collator = make_supervised_data_module(
                tokenizer, data_args, model.config.model_type)

        if local_rank != -1:
            dataset.data = dataset.data.iloc[local_rank::world_size]
        if resume and output_path.exists():
            if resume_exclude_all:
                paths = output_dir.glob('*.jsonl')
                dfs_temp = []
                for path in paths:
                    df_temp = pd.read_json(path, lines=True)
                    dfs_temp.append(df_temp)
                df_resume = pd.concat(dfs_temp)
            else:
                df_resume = pd.read_json(output_path, lines=True)
            assert 'qid' in df_resume.columns, \
                f'Column qid not found in {output_path}'
            dataset.data = dataset.data[~dataset.data['qid'].isin(
                df_resume['qid'])]
        else:
            output_path.write_text('')  # clear

        if len(dataset) == 0:
            logger.warning('No data to inference')
            return

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                collate_fn=data_collator,
                                num_workers=num_workers)

        results = []
        step = 0
        with torch.no_grad():
            for data in tqdm(dataloader):
                input_ids = data['input_ids']
                labels = data['labels']
                input_ids = input_ids.to(model.device)
                output = model(input_ids=input_ids, return_dict=True)
                logits = output.logits
                labels = labels.to('cpu')
                logits = logits.to('cpu')
                scores = logits.softmax(dim=-1)
                preds = logits.argmax(dim=-1)

                if 'qid' in data:
                    qids = data['qid']
                    data_groups = data['data_group']
                else:
                    qids = list(range(len(input_ids)))
                    data_groups = [0] * len(input_ids)

                df_temp = pd.DataFrame({
                    'qid': qids,
                    'data_group': data_groups,
                    'label': list(labels.numpy()),
                    'score': list(scores.numpy()),
                    'pred': list(preds.numpy()),
                })
                df_temp['model_name'] = model_args.model_name_or_path

                for idx, row in df_temp.iterrows():
                    result = row.to_dict()
                    results.append(result)

                if (split_dump_size is not None) \
                    and (split_dump_size > 0) \
                        and (len(results) >= split_dump_size):
                    if output_path is not None:
                        dump_generate_result(results, output_path)
                        step += 1
                        results = []

            if len(results) > 0:
                if output_path is not None:
                    dump_generate_result(results, output_path)
                    step += 1
                    results = []


if __name__ == '__main__':
    fire.Fire(ClassificationRunner)
