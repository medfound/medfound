#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Subset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint
from medfound.models.load_model import load_model
from medfound.utils import setup_logging, export_trainer_model


logger = logging.getLogger(__name__)


@dataclass
class TaskArguments:
    """
    Arguments pertaining to what data we are going to input our model
    for training and eval.
    """

    task_config: Optional[dict] = field(
        default=None,
        metadata={"help": "The task config."},
    )
    lm_type: Optional[str] = field(
        default='dec',
        metadata={"help": "The language model type."},
    )
    share_storage: bool = field(
        default=False,
        metadata={"help": "Whether to share storage in data loading."},
    )


class TrainerFramework(object):
    """A trainer framework."""
    def __init__(self, task=None, trainer_kargs={}, trainer_class=None,
                 make_supervised_data_module=None,
                 model_args_class=None, data_args_class=None,
                 detect_anomaly=False):
        super(TrainerFramework, self).__init__()
        self.task = task
        self.trainer_kargs = trainer_kargs
        self.trainer_class = trainer_class
        self.make_supervised_data_module = make_supervised_data_module
        self.model_args_class = model_args_class
        self.data_args_class = data_args_class
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        """

        Parameters
        ----------
        task: str
            task name
            - lm: language model
        config
        kargs

        Returns
        -------

        """
    def train(self, config=None, **kargs):
        """train

        Args:
            config: Configuration directory. Defaults to None.

        Raises:
            ValueError: Output directory already exists and is not empty.
            ValueError: Unknown task.
            ValueError: Unknown task.
            ValueError: Unknown task.
        """
        task = self.task

        parser = HfArgumentParser((self.model_args_class, self.data_args_class,
                                   TrainingArguments, TaskArguments))
        kargs['remove_unused_columns'] = False
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
        model_args, data_args, training_args, task_args = \
            parser.parse_dict(kargs)
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_level = training_args.get_process_log_level()
        setup_logging(logger, log_level)

        # save model args
        with open(Path(training_args.output_dir)/'model_args.json', 'w') as f:
            json.dump(vars(model_args), f, indent=2)

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, "
            f"device: {training_args.device}, "
            f"n_gpu: {training_args.n_gpu}, "
            f"distributed: {bool(training_args.local_rank != -1)}, "
            f"fp16: {training_args.fp16}, "
            f"bf16: {bool(training_args.bf16)}, "
            f"tf32: {bool(training_args.tf32)}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) \
                and training_args.do_train \
                and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and \
                    len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) "
                    "already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and \
                    training_args.resume_from_checkpoint is None:
                logger.info(
                    "Checkpoint detected, resuming training at "
                    f"{last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to "
                    "train from scratch."
                )

        # Set seed before initializing model.
        set_seed(training_args.seed)

        # Load pretrained model and tokenizer
        if task in ['lm', 'clf']:
            model, tokenizer = load_model(
                task, model_args, model_args.local_dir,
                gradient_checkpointing=training_args.gradient_checkpointing,
                task_config=task_args.task_config)
        else:
            raise ValueError(f'unknown task {task}')
        if model_args.lora:
            def print_peft_model(model_peft, name):
                if hasattr(model_peft, 'peft_config'):
                    print(f'{name} peft_config: ')
                    model_peft.print_trainable_parameters()
            print_peft_model(model, 'model')
            if hasattr(model, 'model_language'):
                print_peft_model(model.model_language, 'model_language')

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # load datasets
        train_dataset = None
        eval_dataset = None
        if training_args.do_train or training_args.do_eval:
            with training_args.main_process_first(
                    desc="data loading", local=not task_args.share_storage):
                if task in ['lm', 'clf']:
                    train_dataset, eval_dataset, data_collator = \
                        self.make_supervised_data_module(
                            tokenizer, data_args, model.config.model_type,
                            lm_type=task_args.lm_type)
                else:
                    raise ValueError(f'unknown task {task}')
            if data_args.max_train_samples is not None:
                max_train_samples = min(
                    len(train_dataset), data_args.max_train_samples)
                train_dataset = Subset(train_dataset,
                                       indices=list(range(max_train_samples)))

            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset),
                                       data_args.max_eval_samples)
                eval_dataset = Subset(eval_dataset,
                                      indices=list(range(max_eval_samples)))

        trainer = self.trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **self.trainer_kargs,
        )
        trainer.label_names = ['labels']
        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

            if not model_args.lora:
                trainer.save_model()

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        kwargs = {}
        if data_args.data_path is not None:
            kwargs["dataset"] = f"dataset_train: {data_args.data_path}, "
            f"dataset_test: {data_args.data_path_test}"

        if training_args.do_train:
            is_load_in_kbit = model_args.load_in_8bit

            if task in ['lm', 'clf']:
                export_trainer_model(trainer, training_args.output_dir,
                                     use_lora=model_args.lora,
                                     load_in_kbit=is_load_in_kbit,
                                     max_shard_size="4GB")
            else:
                raise ValueError(f'unknown task {task}')

    def export(self, output_dir, config=None, resume_from_checkpoint=None,
               **kargs):
        """Export the model.

        Args:
            output_dir (str): Directory to export the model and configuration.
            config (str, optional): Configuration directory. Defaults to None.
            resume_from_checkpoint (str, optional): Path to a checkpoint file
            from which exporting should resume. Defaults to None.
        """
        from medfound.utils import export
        task = self.task
        export(output_dir, task, config, resume_from_checkpoint, **kargs)
