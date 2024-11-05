#!/usr/bin/env python
# coding=utf-8
import warnings
from collections import defaultdict
from typing import Literal, Tuple, Dict, Optional, List
import torch
from deepspeed import DeepSpeedEngine
from peft import PeftModel
from transformers.modeling_utils import unwrap_model
from medfound.data.utils import IGNORE_INDEX
from medfound.models.load_model import ModelArguments
import logging
import fire
from transformers import Trainer
from medfound.utils import T5_MODEL_TYPES
from medfound.trainer import TrainerFramework
import torch.nn.functional as F
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="bitsandbytes.cuda_setup")


logger = logging.getLogger(__name__)


def make_supervised_data_module(tokenizer, data_args, model_type,
                                lm_type=None):
    """
    Generate supervised data module for training.

    Args:
        tokenizer: The tokenizer to use for tokenization.
        data_args: The arguments for data processing.
        model_type (str): Type of the model.
        lm_type (str, optional): Type of language model. Defaults to None.

    Returns:
        Tuple[DPODataset, DPODataset,
        DataCollatorForDPODataset]
    """
    from medfound.data.dpo_dataset import DPODataset
    from medfound.data.dpo_dataset import DataCollatorForDPODataset

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
        train_dataset = DPODataset(tokenizer=tokenizer,
                                   data_path=data_args.data_path,
                                   lm_type=lm_type,
                                   block_size=block_size,
                                   prompt_name=data_args.prompt_name,
                                   add_qid=data_args.add_qid,
                                   cache_dir=data_args.cache_dir,
                                   num_proc=data_args.num_proc)

    eval_dataset = None
    if data_args.data_path_test:
        eval_dataset = DPODataset(tokenizer=tokenizer,
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
    data_collator = DataCollatorForDPODataset(tokenizer=tokenizer,
                                              add_causal_mask=add_causal_mask)
    return train_dataset, eval_dataset, data_collator


compute_metrics = None
preprocess_logits_for_metrics = None


def get_log_probs(logits, labels):
    """Compute the logic probabilities of the given logits and labels.

    Args:
        logits: The input logits.
        labels: The target labels.

    Returns:
        The computed logic probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def masked_mean(data, mask, dim=None, eps=1e-8):
    """Compute the mean of the masked input.

    Args:
        data: The input data.
        mask: The mask to be applied to the input data.
        dim (int, optional): The dimension along which to compute the mean.
        Defaults to None.
        eps (float, optional): A small value to avoid division by zero.
        Defaults to 1e-8.

    Returns:
       The computed mean of the data.
    """
    data = data * mask
    if dim is not None:
        return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
    else:
        return data.sum() / (mask.sum() + eps)


def get_entropy(logits, mask):
    """Compute the entropy of the given logits and mask.

    Args:
        logits: The input logits.
        mask: The input mask.

    Returns:
        The computed entropy of the logits and mask.
    """
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
    return entropy


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = dict()

        self.average_log_prob = False
        self.dpo_beta = 1
        self.reference_free = False
        self.loss_type = "sigmoid"
        self.label_smoothing = 0.0
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def get_model_output(self, model, inputs, average_log_prob=False,
                         is_ref_model=False):
        """Get the output of model.

        Args:
            model: The model used for prediction .
            inputs : The input data for prediction.
            average_log_prob (bool, optional): Whether to compute the average
            logic probabilities. Defaults to False.
            is_ref_model (bool, optional): Whether the model is a reference
            model. Defaults to False.

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """
        input_ids_accepted = inputs["input_ids_accepted"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_accepted = inputs["attention_mask_accepted"]
        attention_mask_rejected = inputs["attention_mask_rejected"]
        labels_accepted = inputs["labels_accepted"]
        labels_rejected = inputs["labels_rejected"]

        if is_ref_model:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                with model.module.disable_adapter():
                    accepts_logits = model(
                        input_ids=input_ids_accepted,
                        attention_mask=attention_mask_accepted,
                        return_dict=True).logits
                    rejects_logits = model(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                        return_dict=True).logits
            elif isinstance(model, DeepSpeedEngine):
                with unwrap_model(model).disable_adapter():
                    accepts_logits = model(
                        input_ids=input_ids_accepted,
                        attention_mask=attention_mask_accepted,
                        return_dict=True).logits
                    rejects_logits = model(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                        return_dict=True).logits
            elif isinstance(model, PeftModel):
                with model.disable_adapter():
                    accepts_logits = model(
                        input_ids=input_ids_accepted,
                        attention_mask=attention_mask_accepted,
                        return_dict=True).logits
                    rejects_logits = model(
                        input_ids=input_ids_rejected,
                        attention_mask=attention_mask_rejected,
                        return_dict=True).logits
            else:
                raise AttributeError(
                    f" model object [{model.__class__.__name__}] has "
                    "no attribute [disable_adapter] "
                )
        else:
            accepts_logits = model(input_ids=input_ids_accepted,
                                   attention_mask=attention_mask_accepted,
                                   return_dict=True).logits
            rejects_logits = model(input_ids=input_ids_rejected,
                                   attention_mask=attention_mask_rejected,
                                   return_dict=True).logits

        accepts_labels, rejects_labels = labels_accepted, labels_rejected
        accepts_action_masks = accepts_labels.ne(IGNORE_INDEX).long()
        rejects_action_masks = rejects_labels.ne(IGNORE_INDEX).long()

        accepts_log_probs = get_log_probs(accepts_logits[:, :-1, :],
                                          input_ids_accepted[:, 1:])
        rejects_log_probs = get_log_probs(rejects_logits[:, :-1, :],
                                          input_ids_rejected[:, 1:])

        if average_log_prob:
            accepts_logps = masked_mean(accepts_log_probs,
                                        accepts_action_masks[:, 1:], dim=-1)
            rejects_logps = masked_mean(rejects_log_probs,
                                        rejects_action_masks[:, 1:], dim=-1)
        else:
            accepts_logps = (accepts_log_probs
                             * accepts_action_masks[:, 1:]).sum(dim=-1)
            rejects_logps = (rejects_log_probs
                             * rejects_action_masks[:, 1:]).sum(dim=-1)

        accepts_entropy = get_entropy(accepts_logits[:, :-1, :],
                                      accepts_action_masks[:, 1:])
        rejects_entropy = get_entropy(rejects_logits[:, :-1, :],
                                      rejects_action_masks[:, 1:])
        return accepts_entropy, rejects_entropy, accepts_logps, rejects_logps

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log
           probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the
            chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for
            the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model
            for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model
            for the rejected responses. Shape: (batch_size,)
            reference_free: If True, we ignore the _provided_ reference model
            and implicitly use a reference model that assigns equal probability
            to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards,
            rejected_rewards).
            The losses tensor contains the DPO loss for each example in the
            batch.
            The chosen_rewards and rejected_rewards tensors contain the
            rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically
        # something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing
        # parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (-F.logsigmoid(self.dpo_beta * logits)
                      * (1 - self.label_smoothing)
                      - F.logsigmoid(-self.dpo_beta * logits)
                      * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.dpo_beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter
            # for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.dpo_beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps
                         - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps
                           - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = \
                policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected)
            # is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.dpo_beta * (chosen_logratios
                                                   - rejected_KL)),
                    1 - F.sigmoid(self.dpo_beta * (chosen_KL
                                                   - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. "
                "Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.dpo_beta
            * (
                policy_chosen_logps.to(self.accelerator.device)
                - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.dpo_beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        inputs,
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Compute the DPO loss and other metrics for the given batch of
        inputs for train or test.
        """
        metrics = {}
        accepts_entropy, rejects_entropy, accepts_logps, rejects_logps = \
            self.get_model_output(model, inputs,
                                  average_log_prob=self.average_log_prob)
        with torch.no_grad():
            _, _, ref_accepts_logps, ref_rejects_logps = \
                self.get_model_output(model, inputs,
                                      average_log_prob=self.average_log_prob,
                                      is_ref_model=True)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            accepts_logps,
            rejects_logps,
            ref_accepts_logps,
            ref_rejects_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards -
                                               rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = rejects_logps.detach().\
            mean().cpu()
        metrics[f"{prefix}logps/chosen"] = accepts_logps.detach().mean().cpu()
        return losses.mean(), metrics

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
        loss, metrics = self.get_batch_loss_metrics(model, inputs,
                                                    train_eval="train")
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def store_metrics(self, metrics: Dict[str, float],
                      train_eval: Literal["train", "eval"] = "train") -> None:
        """Store the metrics.

        Args:
            metrics (Dict[str, float]): A dictionary containing the metrics.
            train_eval (optional): Whether the metrics are for training or
            evaluation. Defaults to "train".
        """
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        """Performs a prediction step using the model.

        Args:
            model: The model used for prediction.
            inputs: The input data for prediction.
            prediction_loss_only (bool): Whether to return only the prediction
            loss.
            ignore_keys (Optional[List[str]], optional): A list of keys to
            ignore in the prediction. Defaults to None.

        Returns:
            tuple: A tuple containing prediction loss, logits, and labels.
        """
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config,
                                      "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs,
                                                        train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0)
                       for k, v in logits_dict.items()
                       if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored
        metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)


class DPORunner(object):
    def __init__(self):
        from medfound.data.sft_dataset import DataArguments
        trainer_kargs = dict(
        )
        self.trainer = TrainerFramework(
            task='lm', trainer_kargs=trainer_kargs,
            trainer_class=CustomTrainer,
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


if __name__ == '__main__':
    fire.Fire(DPORunner)
