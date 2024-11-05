import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import torch
import transformers
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, \
    CONFIG_MAPPING, AutoTokenizer, AutoModelForCausalLM, \
    AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, \
    BitsAndBytesConfig, PreTrainedTokenizer
from medfound.models.peft import wrap_peft_model
from medfound.utils import T5_MODEL_TYPES

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to
    fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if"
                " you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from"
                  " the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model"
                " is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,"
                "summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if "
                                "not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if "
                                "not the same as model_name"}
    )
    local_files_only: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models "
                  "downloaded from huggingface.co"},
    )
    local_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models "
                  "downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer "
                  "(backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch "
                  "name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running "
                "`huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in 8-bit precision "
                  "(fp16)."},
    )
    load_in_cuda: bool = field(
        default=False,
        metadata={"help": "Whether to load the model in CUDA."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under "
                "this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    model_compile: bool = field(
        default=False,
        metadata={"help": "Whether to compile the model with PyTorch 2.0."},
    )

    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LORA."},
    )
    lora_config: str = field(
        default="",
        metadata={"help": "LORA config path."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LORA r."},
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "LORA target modules."},
    )
    lora_modules_to_save: str = field(
        default=None,
        metadata={"help": "LORA modules to save. List of modules apart from "
                  "LoRA layers to be set as trainable and saved in the final "
                  "checkpoint."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LORA alpha."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LORA dropout."},
    )

    def __post_init__(self):
        if (
            self.config_overrides is not None
            and (self.config_name is not None
                 or self.model_name_or_path is not None)):
            raise ValueError(
                "--config_overrides can't be used in combination with "
                "--config_name or --model_name_or_path"
            )
        if isinstance(self.lora_target_modules, str):
            if self.lora_target_modules == '':
                self.lora_target_modules = None
            else:
                self.lora_target_modules = self.lora_target_modules.split(',')
        if isinstance(self.lora_modules_to_save, str):
            if self.lora_modules_to_save == '':
                self.lora_modules_to_save = None
            else:
                self.lora_modules_to_save = \
                    self.lora_modules_to_save.split(',')


def load_tokenizer(model_name_or_path=None, tokenizer_name=None,
                   use_auth_token=False, local_dir=None,
                   use_fast_tokenizer=True, model_revision=None):
    """Load a tokenizer for given a model name or tokenizer name

    Args:
        model_name_or_path (str, optional): The model name or path.
        Defaults to None.
        tokenizer_name (str, optional): The tokenizer name. Defaults to None.
        use_auth_token (bool, optional): Whether to use authentication token
        for accessing Hugging Face Hub. Defaults to False.
        local_dir (str, optional): Local directory to cache models and
        tokenizers. Defaults to None.
        use_fast_tokenizer (bool, optional): Whether to use fast tokenizer.
        Defaults to True.
        model_revision (str, optional): Revision of the model.
        Defaults to None.

    Raises:
        ValueError: If neither model_name_or_path nor tokenizer_name is
        provided.

    Returns:
        A tokenizer instance
    """
    tokenizer_kwargs = {
        "cache_dir": local_dir,
        "use_fast": use_fast_tokenizer,
        "revision": model_revision,
        "trust_remote_code": True,
        "use_auth_token": True if use_auth_token else None,
    }
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  **tokenizer_kwargs)
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. "
            "This is not supported by this script."
            "You can do it from another script, save it, "
            "and load it from here, using --tokenizer_name."
        )

    return tokenizer


def load_model(task, model_args, local_dir=None, gradient_checkpointing=False,
               tokenizer_only=False, task_config={}):
    """Load pretrained model and tokenizer

    Args:
        task (str): The task of model to be loaded.
        Supported tasks: 'lm', 'clf'.
        model_args: Various arguments related to the model.
        local_dir (str, optional): Local directory path to cache model and
        tokenizer. Defaults to None.
        gradient_checkpointing (bool, optional): Whether to use
        gradient_checkpointing. Defaults to False.
        tokenizer_only (bool, optional): Whether to load only the tokenizer.
        Defaults to False.
        task_config (dict, optional): Extra config. Defaults to {}.

    Raises:
        NotImplementedError: Unsuppored tasks.

    Returns:
        Union[Tuple[Model, Tokenizer], Model]:
            - A tuple containing the model instance and tokenizer instance.
            - A tokenizer instance.
    """

    if task_config is None:
        task_config = {}
    elif isinstance(task_config, str):
        task_config = json.loads(Path(task_config).read_text())

    config_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": True,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if gradient_checkpointing:
        config_kwargs["gradient_checkpointing"] = True
        config_kwargs["use_cache"] = False
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name,
                                            **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                                            **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer = load_tokenizer(
        model_args.model_name_or_path, model_args.tokenizer_name,
        model_args.use_auth_token, model_args.local_dir,
        model_args.use_fast_tokenizer, model_args.model_revision)

    if tokenizer_only:
        return tokenizer

    if task == "lm":  # **task related
        AutoModel = AutoModelForCausalLM
        if config.model_type in T5_MODEL_TYPES:
            AutoModel = AutoModelForSeq2SeqLM
    elif task == "clf":
        AutoModel = AutoModelForSequenceClassification
        if 'labels' in task_config:
            labels = task_config['labels']
            config.num_labels = len(labels)

        if hasattr(config, 'problem_type'):
            config.problem_type = 'single_label_classification'
    else:
        raise NotImplementedError(f"task {task} not implemented")

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        args_extra = dict()

        if model_args.load_in_8bit:
            args_extra["load_in_8bit"] = model_args.load_in_8bit

        if model_args.load_in_cuda:
            args_extra["device_map"] = {"": int(os.environ.get(
                "LOCAL_RANK", "0"))}

        model = AutoModel.from_pretrained(  # **task related
            Path(model_args.model_name_or_path),
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=local_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **args_extra,
        )

    else:
        model = AutoModel.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel())
                            for p in model.parameters()).values())
        logger.info("Training new model from scratch - Total size="
                    f"{n_params / 2 ** 20:.2f}M params")

    if model_args.lora:
        try:
            # If device_map exists, it will automatically align.
            model = wrap_peft_model(
                model, model_args, model_args.lora_target_modules,
                'language model')
        except Exception as e:
            print(e)
            if 'not found in the base model.' in str(e):
                print(f'Language model: {e}')

    if model_args.model_compile:
        model = torch.compile(model)

    return model, tokenizer
