## Description
The accurate diagnosis is crucial in healthcare. Here, we introduce MedFound, which is a medical large language model (Medical LLM) pretrained on medical text and real-world clinical records. We fine-tuned MedFound using a self-bootstrapping strategy to learn diagnostic reasoning and incorporated a preference alignment framework to align with standard clinical practice. Our approach results in MedFound-DX-PA, a LLM based diagnostic system that aligns with clinical requirements. This repository contains the code used for data preprocessing, model development, and evaluation in our study (A Generalist Medical Language Model for Disease Diagnosis Assistance).

## Code Structure
- `config/`: Configuration files for training and evaluation
- `medfound/`: Main source code folder
  - `data/`: Contains the implementation of the data pipeline, including data loading and preprocessing
    - `clf_dataset.py`: Data pipeline for classification.
    - `dpo_dataset.py`: Data pipeline for direct preference optimization.
    - `io.py`: IO related functions, such as reading and exporting.
    - `pretrain_dataset.py`: Data pipeline for pretraining.
    - `prompt.py`: Prompting preprocessing functions.
    - `sft_dataset.py`: Data pipeline for fine-tuning.
    - `utils.py`: The utility functions for data pipeline.
  - `models/`: Contains the implementation related to models.
    - `load_model.py`: Loading models.
    - `peft.py`: parameter-efficient fine-tuning.
  - `trainer.py`: Training and validation framework.
  - `utils.py`: The utility functions
- `run_clf.py`: The script for classification.
- `run_clm.py`: The script for fine-tuning.
- `run_dpo.py`: The script for direct preference optimization.
- `run_pretrain.py`: The script for pretraining.
- `requirements.txt`: List of required dependencies
- `README.md`: This README file

## Run Demo
The model for the demo can be downloaded from [Huggingface](https://huggingface.co/medicalai/MedFound-Llama3-8B-finetuned). More models can be found here: [MedFound-7B](https://huggingface.co/medicalai/MedFound-7B), [MedFound-Llama3-8B-finetuned](https://huggingface.co/medicalai/MedFound-Llama3-8B-finetuned), [MedFound-R1-Qwen-7B-EN-preview](https://huggingface.co/medicalai/MedFound-R1-Qwen-7B-EN-preview) and [MedFound-176B](https://huggingface.co/medicalai/MedFound-176B).

```python
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "medicalai/MedFound-R1-Qwen-7B-CN-preview"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
data = pd.read_json('data/test.zip', lines=True).iloc[1]
prompt = f"{data["context"]}\n\nPlease provide a detailed and comprehensive diagnostic analysis of this medical record, and give the diagnostic results.\n"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
input_ids = tokenizer([text], return_tensors="pt").to(model.device)
output_ids = model.generate(**input_ids, max_new_tokens=2048, temperature=0.7, do_sample=True).to(model.device)
generated_text = tokenizer.decode(output_ids[0,len(input_ids[0]):], skip_special_tokens=True)
print("Generated Output:\n", generated_text)
```

## Citation
Please cite this article:  
Wang, G., Liu, X., Liu, H., Yang, G. et al. A Generalist Medical Language Model for Disease Diagnosis Assistance. Nat Med (2025). https://doi.org/10.1038/s41591-024-03416-6
