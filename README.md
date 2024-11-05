# MedFound
## Description
The accurate diagnosis is crucial in healthcare. Here, we introduce MedFound, which is a medical large language model (Medical LLM) pretrained on medical text and real-world clinical records. We fine-tuned MedFound using a self-bootstrapping strategy to learn diagnostic reasoning and incorporated a preference alignment framework to align with standard clinical practice. Our approach results in MedFound-DX-PA, a LLM based diagnostic system that aligns with clinical requirements. This repository contains the code used for data preprocessing, model development, and evaluation in our study (Development and evaluation of diagnostic generalist cross specialties using medical large language model).

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