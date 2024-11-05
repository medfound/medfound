#!/usr/bin/env python
# coding=utf-8
import warnings
from medfound.models.load_model import ModelArguments
import logging
import fire
from transformers import (
    Trainer,
)
from medfound.trainer import TrainerFramework

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="bitsandbytes.cuda_setup")


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
        Tuple[PretrainDataset, PretrainDataset,
        DataCollatorForPretrainDataset]
    """
    from medfound.data.pretrain_dataset import PretrainDataset
    from medfound.data.pretrain_dataset import DataCollatorForPretrainDataset

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 4096:
            logger.warning(
                "The tokenizer picked seems to have a very large "
                f"`model_max_length` ({tokenizer.model_max_length}). "
                "Picking 4096 instead. You can change that default value "
                "by passing --block_size xxx."
            )
            block_size = 4096
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is "
                "larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using "
                f"block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    train_dataset = None
    if data_args.data_path:
        train_dataset = PretrainDataset(tokenizer=tokenizer,
                                        data_path=data_args.data_path,
                                        block_size=block_size,
                                        cache_dir=data_args.cache_dir,
                                        num_proc=data_args.num_proc)
    if data_args.data_path_test:
        eval_dataset = PretrainDataset(tokenizer=tokenizer,
                                       data_path=data_args.data_path_test,
                                       block_size=block_size,
                                       cache_dir=data_args.cache_dir,
                                       num_proc=data_args.num_proc)
    else:
        eval_dataset = train_dataset

    if model_type in ['glm', 'chatglm', 'cpmant']:
        add_causal_mask = True
    else:
        add_causal_mask = False
    data_collator = DataCollatorForPretrainDataset(
        tokenizer=tokenizer,
        add_causal_mask=add_causal_mask
        )
    return train_dataset, eval_dataset, data_collator


compute_metrics = None
preprocess_logits_for_metrics = None


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss.

        Args:
            model (PreTrainedModel): The model to compute the loss for.
            inputs: The inputs to the model.
            return_outputs (bool): Whether to return the outputs along with
            the loss. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, ModelOutput]]: The
            computed loss value.
        """
        raise super(CustomTrainer, self).compute_loss(model, inputs,
                                                      return_outputs)


class PretrainRunner(object):
    def __init__(self):
        from medfound.data.pretrain_dataset import DataArguments
        trainer_kargs = dict(
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.trainer = TrainerFramework(
            task='lm', trainer_kargs=trainer_kargs, trainer_class=Trainer,
            make_supervised_data_module=make_supervised_data_module,
            model_args_class=ModelArguments,
            data_args_class=DataArguments)

    def train(self, config=None, **kargs):
        """Train a model.

        Args:
            config (optional): The configuration for training.
            Defaults to None.
        """
        if 'lm_type' not in kargs:
            kargs['lm_type'] = 'dec'  # default: CLM decoder
        self.trainer.train(config=config, **kargs)

    def prepare_data(self, config=None, **kargs):
        """Prepares the data for the model.

        Args:
            config (optional): The configuration for data preparation.
            Defaults to None.
        """
        self.trainer.prepare_data(config=config, **kargs)

    def export(self, output_dir, config=None,
               resume_from_checkpoint=None, **kargs):
        """Export a model

        Args:
            output_dir (str or Path): Output directory.
            task (str): Task name.
            config (optional): Configuation. Defaults to None.
            resume_from_checkpoint (str, optional): Checkpoint path.
            Defaults to None.
        """
        self.trainer.export(output_dir, config=config,
                            resume_from_checkpoint=resume_from_checkpoint,
                            **kargs)


if __name__ == '__main__':
    fire.Fire(PretrainRunner)
