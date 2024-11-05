import logging
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import datasets
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional
from medfound.data.utils import IGNORE_INDEX, pad_first, pad_last, \
    tokenized_dec_func, tokenized_encdec_func
from medfound.data.prompt import SftPrompter


logger = logging.getLogger("datasets.arrow_dataset")
logger.setLevel(logging.ERROR)


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


def raw_to_formatted(example, prompter=None):
    """Format raw data

    Args:
        example (dict): The raw example data.
        prompter (MultiPrompter, optional): Prompter object for generating
        prompts. Defaults to None.

    Returns:
        dict: The formatted example data.
    """
    if prompter is None:
        prompter = SftPrompter()
    texts = example['text'].split('[[SEP]]')
    texts = prompter.add_prefix(texts)
    result = {
        'texts': texts,
    }
    if 'trainable_round' in example:
        result['trainable_round'] = example['trainable_round']
    return result


class SupervisedDataset(Dataset):
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
        lm_type (str): Type of the language model.
        block_size (int, optional): The maximum sequence length allowed after
        tokenization. Defaults to None.
        cache_dir (str, optional): Directory to cache the processed data.
        Defaults to None.
        num_proc (int, optional): Number of processes for parallel processing.
        Defaults to None.
        prompt_name (str, optional): Name of the prompt. Defaults to None.
        add_qid (bool, optional): Whether to add query ids. Defaults to False.
    """

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, lm_type,
                 block_size=None, cache_dir: str = None, num_proc: int = None,
                 prompt_name=None, add_qid=False, **kargs):
        """

        Parameters
        ----------
        data_path
        tokenizer
        lm_type: str
            - dec: decoder type.
            - encdec: encoder-decoder type.

        block_size
        cache_dir
        num_proc
        max_data_length
        prompt_name
        add_qid
        """
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data...")
        prompter = SftPrompter(prompt_name)

        # Multi data concatenation
        if isinstance(data_path, tuple):
            data_paths = list(data_path)
        elif isinstance(data_path, Path):
            data_paths = [str(data_path)]
        else:
            data_paths = data_path.split(',')

        if lm_type == "dec":
            tokenized_func = tokenized_dec_func
        elif lm_type == "encdec":
            tokenized_func = tokenized_encdec_func
        else:
            raise ValueError(f"Unexpected lm_type: {lm_type}")

        cols_valid = ['text', 'qid', 'trainable_round']
        dfs_data = []
        for data_path in data_paths:
            dataset_jsonl = datasets.load_dataset(
                'json', data_files=data_path, cache_dir=cache_dir)
            dataset_jsonl = dataset_jsonl['train']
            cols_select = list(
                set(dataset_jsonl.column_names) & set(cols_valid))
            dataset_jsonl = dataset_jsonl.select_columns(cols_select)
            # Filter out empty values
            dataset_jsonl = dataset_jsonl.filter(
                lambda x: x['text'] is not None, num_proc=num_proc)

            dataset_jsonl = dataset_jsonl.map(
                raw_to_formatted, fn_kwargs={'prompter': prompter},
                num_proc=num_proc)

            dataset_jsonl = dataset_jsonl.filter(
                lambda x: sum([t == '' for t in x['texts'][1::2]]) == 0,
                num_proc=num_proc)

            dataset_jsonl = \
                dataset_jsonl.map(self._tokenize, fn_kwargs=dict(
                    tokenizer=tokenizer, block_size=block_size,
                    tokenized_func=tokenized_func),
                                  num_proc=num_proc)
            if lm_type == "dec":
                dataset_jsonl = dataset_jsonl.filter(
                    lambda x: (len(x['input_ids']) < block_size),
                    num_proc=num_proc)  # Filter length
            elif lm_type == "encdec":
                dataset_jsonl = dataset_jsonl.filter(
                    lambda x: len(x['input_ids']) +
                    len(x['labels']) < block_size, num_proc=num_proc)

            df_data = dataset_jsonl.to_pandas()
            df_data['data_group'] = Path(data_path).name
            dfs_data.append(df_data)

        df_data = pd.concat(dfs_data, ignore_index=True)

        self.block_size = block_size
        self.tokenizer = tokenizer
        self.tokenized_func = tokenized_func
        self.add_qid = add_qid
        self.data = df_data

    def _tokenize(self, example, tokenizer, block_size, tokenized_func):
        texts = example['texts']
        n_turn = len(texts) // 2
        if 'trainable_round' in example:
            trainable_round = example['trainable_round']
            if trainable_round == -1:  # Train only the last round
                i_turn = n_turn
            else:
                # Randomly select a location as the conversation endpoint
                i_turn = np.random.randint(0, n_turn) + 1
        else:
            # Randomly select a location as the conversation endpoint
            i_turn = np.random.randint(0, n_turn) + 1
        texts = texts[:i_turn * 2]
        source = ''.join(texts[:-1])
        target = texts[-1]+tokenizer.eos_token  # add eos to labels

        # tokenized
        example = {'source': source, 'target': target}
        example_tokenized = tokenized_func(
            example, tokenizer, block_size=block_size*2, add_eos=False)
        return example_tokenized

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.data.iloc[i]
        example_tokenized = example

        input_ids = example_tokenized["input_ids"]
        labels = example_tokenized["labels"]
        if (self.block_size is not None) and \
                (len(input_ids) > self.block_size):
            # truncature from back to front
            input_ids = input_ids[-self.block_size:]
            labels = labels[-self.block_size:]

        sample = {
            'input_ids': input_ids.long(),
            'labels': labels.long(),
            'attention_mask': input_ids.long().ne(self.tokenizer.pad_token_id),
        }
        if self.add_qid:
            sample['qid'] = example['qid']
            sample['data_group'] = example['data_group']
        return sample


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning..

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for tokenization.
        add_causal_mask (bool, optional): Whether to add causal mask.
        Defaults to False.
        padding_side (str, optional): The side for padding.
        Defaults to 'right'.

    """
    tokenizer: transformers.PreTrainedTokenizer
    add_causal_mask: bool = False
    padding_side: str = 'right'

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "attention_mask"))
        if self.padding_side == 'left':
            input_ids = pad_first(input_ids, self.tokenizer.pad_token_id)
            labels = pad_first(labels, IGNORE_INDEX)
            attention_mask = pad_first(attention_mask, 0)
        else:
            input_ids = pad_last(input_ids, self.tokenizer.pad_token_id)
            labels = pad_last(labels, IGNORE_INDEX)
            attention_mask = pad_last(attention_mask, 0)
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if "qid" in instances[0]:
            qid = [instance["qid"] for instance in instances]
            data["qid"] = qid
        if "data_group" in instances[0]:
            data_group = [instance["data_group"] for instance in instances]
            data["data_group"] = data_group
        if self.add_causal_mask:
            # Upper triangular matrix
            attention_mask = np.triu(np.ones((
                input_ids.shape[1], input_ids.shape[1]), dtype=np.bool_), k=1)
            attention_mask = np.repeat(attention_mask[np.newaxis, :, :],
                                       input_ids.shape[0], axis=0)
            attention_mask = torch.from_numpy(attention_mask)
            data["attention_mask"] = attention_mask

        return data
