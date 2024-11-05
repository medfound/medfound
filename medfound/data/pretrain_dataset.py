import itertools
import logging
from dataclasses import dataclass, field

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional
from medfound.data.utils import IGNORE_INDEX, pad_last


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
    prompt_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the prompt to use (via the prompt library)."
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

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models "
                  "downloaded from huggingface.co"},
    )

    num_proc: Optional[int] = field(
        default=32,
        metadata={"help": "Number of processes to use for data loading."},
    )

    add_qid: bool = field(
        default=False,
        metadata={"help": "Whether to add qid to the input_ids."},
    )


def tokenized_func(example, tokenizer: transformers.PreTrainedTokenizer):
    """Preprocess the data by tokenizing.

    Args:
        example (dict): A dictionary containing the data to be tokenized.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for
        tokenization.

    Returns:
        Dict: A dictionary containing the tokenized input ids.
    """
    result = tokenizer(example['text'], padding=False)
    return result


def group_texts(examples, block_size):
    """Group texts into blocks of a specified size.

    Args:
        examples (Dict): A dictionary containing texts.
        block_size (int): The desired size for each block.

    Returns:
        Dict: A dictionary containing grouped texts.
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(itertools.chain(*examples[k]))
                             for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop,
    # you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


class PretrainDataset(Dataset):
    """
    Dataset for pretraining.

    1. raw to formatted: Add prefixes and suffixes.
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

    """

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, block_size=None,
                 cache_dir: str = None, num_proc: int = None):
        super(PretrainDataset, self).__init__()

        logging.warning("Loading data...")
        logging.warning("Formatting inputs...")

        data_paths = str(data_path).split(',')
        dfs_data = []
        for path in data_paths:
            data = datasets.load_dataset(
                'text', data_files=path, cache_dir=cache_dir)
            data = data.map(
                tokenized_func, batched=True,
                fn_kwargs={'tokenizer': tokenizer}, num_proc=num_proc)
            data = data.map(group_texts, batched=True, num_proc=num_proc,
                            fn_kwargs={'block_size': block_size})
            data = data.filter(lambda x: (len(x['labels']) > 0),
                               num_proc=num_proc)
            df_data = data['train'].to_pandas()
            dfs_data.append(df_data)
        self.data = pd.concat(dfs_data, ignore_index=True)

        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = {
            'input_ids': torch.Tensor(self.data.iloc[i]['input_ids']).long(),
            'labels': torch.Tensor(self.data.iloc[i]['input_ids']).long(),
        }
        return sample


@dataclass
class DataCollatorForPretrainDataset(object):
    """Collate examples for pretraining.

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
        labels = pad_last(labels, IGNORE_INDEX)
        data = {
            "input_ids": input_ids,
            "labels": labels,
        }
        if self.add_causal_mask:
            attention_mask = np.triu(
                np.ones((input_ids.shape[1], input_ids.shape[1]),
                        dtype=np.bool_), k=1)  # Upper triangular matrix
            attention_mask = np.repeat(
                attention_mask[np.newaxis, :, :], input_ids.shape[0], axis=0)
            attention_mask = torch.from_numpy(attention_mask)
            data["attention_mask"] = attention_mask

        return data
