import logging
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import datasets
import torch
import transformers
from torch.utils.data import Dataset
from typing import Dict, Sequence, Optional
from medfound.data.utils import IGNORE_INDEX, pad_first, pad_last, \
     tokenized_dec_func, tokenized_encdec_func
from medfound.data.prompt import SftPrompter


# ignore warnings when loading datasets from cache
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

    num_proc: Optional[int] = field(
        default=32,
        metadata={"help": "Number of processes to use for data loading."},
    )

    add_qid: bool = field(
        default=False,
        metadata={"help": "Whether to add qid to the input_ids."},
    )


def raw_to_formatted(example, prompter=None):
    """Format raw data.

    Args:
        example (dict): The raw example data. A dictionary containing 'text'
        key.
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


class DPODataset(Dataset):
    """
    Dataset for DPO.
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
                 prompt_name=None, add_qid=False):
        super(DPODataset, self).__init__()

        logging.warning("Loading data...")
        prompter = SftPrompter(prompt_name)

        # multiple data concatenation
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
        cols_valid = ['text', 'qid', 'trainable_round', 'accepted', 'rejected']
        dfs_data = []
        for data_path in data_paths:
            dataset_jsonl = datasets.load_dataset(
                'json', data_files=data_path, cache_dir=cache_dir)
            dataset_jsonl = dataset_jsonl['train']
            cols_select = list(
                set(dataset_jsonl.column_names) & set(cols_valid))
            dataset_jsonl = dataset_jsonl.select_columns(cols_select)
            df_data = dataset_jsonl.to_pandas()
            df_data['data_group'] = Path(data_path).name
            dfs_data.append(df_data)

        df_data = pd.concat(dfs_data, ignore_index=True)

        self.prompter = prompter
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.tokenized_func = tokenized_func
        self.add_qid = add_qid

        self.data = df_data

    def _tokenize(self, example, tokenizer, prompter, block_size,
                  tokenized_func):
        query = example['text'].split('[[SEP]]')[0]

        example_tokenized_all = {}
        for col in ['text', 'accepted', 'rejected']:
            # tokenized
            response = example[col].split('[[SEP]]')[-1]
            # prompt, response =
            texts = prompter.add_prefix([query, response])
            example_str = {
                'source': ''.join(texts[:-1]),
                'target': texts[-1]+tokenizer.eos_token,
            }
            example_tokenized = tokenized_func(
                example_str, tokenizer, block_size=block_size*2, add_eos=False)
            example_tokenized_all[col] = example_tokenized

        return example_tokenized_all

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        example = self.data.iloc[i]
        example_tokenized_all = self._tokenize(
            example, tokenizer=self.tokenizer, prompter=self.prompter,
            block_size=self.block_size, tokenized_func=self.tokenized_func)

        for k in example_tokenized_all.keys():
            input_ids = example_tokenized_all[k]["input_ids"]
            labels = example_tokenized_all[k]["labels"]
            if (self.block_size is not None) and \
                    (len(input_ids) > self.block_size):
                input_ids = input_ids[-self.block_size:]
                labels = labels[-self.block_size:]
            example_tokenized_all[k]["input_ids"] = input_ids
            example_tokenized_all[k]["labels"] = labels

        sample = {
            'input_ids': example_tokenized_all['text']["input_ids"].long(),
            'labels': example_tokenized_all['text']["labels"].long(),
            'attention_mask': example_tokenized_all['text']["input_ids"].
            long().ne(self.tokenizer.pad_token_id),
            'input_ids_accepted': example_tokenized_all['accepted']
            ["input_ids"].long(),
            'input_ids_rejected': example_tokenized_all['rejected']
            ["input_ids"].long(),
            'attention_mask_accepted': example_tokenized_all['accepted']
            ["input_ids"].long().ne(self.tokenizer.pad_token_id),
            'attention_mask_rejected': example_tokenized_all['rejected']
            ["input_ids"].long().ne(self.tokenizer.pad_token_id),
            'labels_accepted': example_tokenized_all['accepted']["labels"].
            long(),
            'labels_rejected': example_tokenized_all['rejected']["labels"].
            long(),
        }
        if self.add_qid:
            sample['qid'] = example['qid']
            sample['data_group'] = example['data_group']
        return sample


@dataclass
class DataCollatorForDPODataset(object):
    """Collate examples for DPO.

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
        data = {}
        keys_inputs = ["input_ids", "input_ids_accepted", "input_ids_rejected"]
        keys_labels = ["labels", "labels_accepted", "labels_rejected"]
        keys_attention_mask = ["attention_mask", "attention_mask_accepted",
                               "attention_mask_rejected"]

        # reorganize
        for key in keys_inputs+keys_labels+keys_attention_mask:
            data[key] = [instance[key] for instance in instances]

        # padding
        if self.padding_side == 'left':
            pad_func = pad_first
        else:
            pad_func = pad_last
        for key in keys_inputs:
            data[key] = pad_func(data[key], self.tokenizer.pad_token_id)
        for key in keys_labels:
            data[key] = pad_func(data[key], IGNORE_INDEX)
        for key in keys_attention_mask:
            data[key] = pad_func(data[key], 0)

        if "qid" in instances[0]:
            qid = [instance["qid"] for instance in instances]
            data["qid"] = qid
        if "data_group" in instances[0]:
            data_group = [instance["data_group"] for instance in instances]
            data["data_group"] = data_group

        return data
