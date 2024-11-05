import re
import warnings

from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel


def wrap_peft_model(model, model_args, lora_target_modules, name='model_name'):
    """Wrap the model with PEFT (Parameter-Efficient Fine-Tuning).

    Args:
        model (Any): The base model to be wrapped.
        model_args (Any): Arguments related to the model.
        lora_target_modules (Any): Target modules for PEFT.
        name (str, optional): Name of the model. Defaults to 'model_name'.

    Returns:
        Any: Model with PEFT functionality wrapped.
    """
    if model_args.lora:
        if lora_target_modules is None or len(lora_target_modules) == 0:
            warnings.warn(f"No target modules specified, {name} will frozen "
                          "all layers")
            for name, param in model.named_parameters():
                # freeze base model's layers
                param.requires_grad = False

            if model_args.lora_modules_to_save is not None:
                module_to_save_names = model_args.lora_modules_to_save
                if isinstance(module_to_save_names, str):
                    module_to_save_names = module_to_save_names.split(',')
                for module_to_save_name in module_to_save_names:
                    for name, param in model.named_parameters():
                        name = name.replace('.weight', '')
                        # unfreeze specified layers
                        if re.match(f".*{module_to_save_name}$", name):
                            param.requires_grad = True
        else:

            if model_args.lora_config:
                peft_config = LoraConfig.from_json_file(model_args.lora_config)
            else:
                lora_config = dict(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=lora_target_modules,
                    modules_to_save=model_args.lora_modules_to_save,
                    inference_mode=False,
                    bias="none",
                    fan_in_fan_out=False
                )
                peft_config = LoraConfig(peft_type="LORA", **lora_config)

            if model_args.load_in_8bit:
                model = prepare_model_for_kbit_training(model)

            model = PeftModel(model=model, peft_config=peft_config)
    return model
