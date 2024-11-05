import logging
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional
from medfound.data.utils import tokenize_fn, IGNORE_INDEX, pad_last


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval.
    """

    data_path: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use "
                                "(via the datasets library)."}
    )
    data_path_test: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use "
                                "(via the datasets library)."}
    )
    cache_dir: Optional[str] = field(
        default='cache', metadata={"help": "Where do you want to store "
                                   "the pre-trained models downloaded from s3"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the "
                "number of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the "
                "number of evaluation examples to this value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size "
                "for training. "
                "Default to the model max input length for single sentence "
                "inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and "
                                 "evaluation sets"}
    )

    num_proc: Optional[int] = field(
        default=32,
        metadata={"help": "Number of processes to use for data loading."},
    )

    add_qid: bool = field(
        default=False,
        metadata={"help": "Whether to add qid to the input_ids."},
    )

    col_text: str = field(
        default='text',  # text
        metadata={"help": "The name of the text column."}
    )
    col_label: str = field(
        default='label',  # label
        metadata={"help": "The name of the label column."}
    )


def raw_to_formatted(example):
    """Format raw data.

    Args:
        example (dict): A dictionary containing 'text' and 'label' keys.

    Returns:
        dict: A dictionary containing the extracted text and the original
        label.
    """
    text = example['text'].split('[[SEP]]')[0]
    label = example['label']
    result = {'text': text, 'label': label}
    return result


def tokenized_func(example, tokenizer: transformers.PreTrainedTokenizer,
                   block_size=None) -> Dict:
    """Preprocess the data by tokenizing.

    Args:
        example (dict): A dictionary containing the data to be tokenized.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for
        tokenization.
        block_size (int, optional): The maximum sequence length allowed after
        tokenization. Defaults to None.

    Returns:
        Dict: A dictionary containing the tokenized input ids.
    """
    text_tokenized = tokenize_fn(example['text'], tokenizer, block_size)
    input_ids = text_tokenized["input_ids"]
    return dict(input_ids=input_ids)


class SupervisedClassificationDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    1. raw to formatted: Add prefix and suffix
        input:
        [(question, response)] -> [(## user: question ## assisant, response)]
    2. tokenized: convert to id
        input:
        [(## user: question ## assisant, response)] -> [(input_ids, labels)]
    3. filter length
        Consider multiple rounds of dialogue and truncate to the maximum
        length position

    Args:
        data_path (str): The path to the data.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for
        tokenization.
        block_size (int, optional): The maximum sequence length allowed after
        tokenization. Defaults to None.
        cache_dir (str, optional): Directory to cache the processed data.
        Defaults to None.
        num_proc (int, optional): Number of processes for parallel processing.
        Defaults to None.
        add_qid (bool, optional): Whether to add query IDs. Defaults to False.
    """
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, block_size=None,
                 cache_dir: str = None, num_proc: int = None, add_qid=False):
        super(SupervisedClassificationDataset, self).__init__()

        logging.warning("Loading data...")

        # multiple data concatenation
        if isinstance(data_path, tuple):
            data_paths = list(data_path)
        else:
            data_paths = data_path.split(',')
        cols_valid = ['text', 'qid', 'label']
        dfs_data = []
        for data_path in data_paths:
            dataset_jsonl = datasets.load_dataset(
                'json', data_files=data_path, cache_dir=cache_dir)
            dataset_jsonl = dataset_jsonl['train']
            cols_select = list(
                set(dataset_jsonl.column_names) & set(cols_valid))
            dataset_jsonl = dataset_jsonl.select_columns(cols_select)
            # Filter null values
            dataset_jsonl = dataset_jsonl.filter(
                lambda x: x['text'] is not None, num_proc=num_proc)
            dataset_jsonl = dataset_jsonl.map(
                raw_to_formatted, num_proc=num_proc)
            df_data = dataset_jsonl.to_pandas()
            df_data['data_group'] = Path(data_path).name
            dfs_data.append(df_data)
        df_data = pd.concat(dfs_data, ignore_index=True)

        self.block_size = block_size
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = block_size
        self.tokenized_func = tokenized_func
        self.add_qid = add_qid
        self.data = df_data

    def _tokenize(self, example):
        example_tokenized = self.tokenized_func({'text': example['text']},
                                                self.tokenizer)
        if 'label' in example:
            example_tokenized['label'] = example['label']
        else:
            example_tokenized['label'] = IGNORE_INDEX
        return example_tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.data.iloc[i]
        example_tokenized = self._tokenize(example)
        input_ids = example_tokenized["input_ids"]
        labels = example_tokenized["label"]
        if (self.block_size is not None) and \
                (len(input_ids) > self.block_size):
            # truncate from front to back
            input_ids = input_ids[:self.block_size]

        sample = {
            'input_ids': input_ids.long(),
            'labels': torch.Tensor([labels]).long(),
        }
        if self.add_qid:
            sample['qid'] = example['qid']
            sample['data_group'] = example['data_group']
        return sample


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for
        tokenization.
        add_causal_mask (bool, optional): Whether to add a causal mask.
        Defaults to False.

    """

    tokenizer: transformers.PreTrainedTokenizer
    add_causal_mask: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = pad_last(input_ids, self.tokenizer.pad_token_id)
        labels = torch.cat(labels, dim=0)
        data = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if "qid" in instances[0]:
            qid = [instance["qid"] for instance in instances]
            data["qid"] = qid
        if "data_group" in instances[0]:
            data_group = [instance["data_group"] for instance in instances]
            data["data_group"] = data_group

        if self.add_causal_mask:
            # Upper triangular matrix
            attention_mask = np.triu(np.ones(
                (input_ids.shape[1], input_ids.shape[1]), dtype=np.bool_), k=1)
            attention_mask = np.repeat(
                attention_mask[np.newaxis, :, :], input_ids.shape[0], axis=0)
            attention_mask = torch.from_numpy(attention_mask)
            data["attention_mask"] = attention_mask

        return data
