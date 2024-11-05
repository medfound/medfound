from typing import Dict, List
import torch
import transformers


def tokenize_fn(text, tokenizer: transformers.PreTrainedTokenizer,
                block_size=None, add_eos=False) -> Dict:
    """Tokenize input text using a given tokenizer.

    Args:
        text (str): The input text to be tokenized.
        tokenizer (transformers.PreTrainedTokenizer): The pre-trained
        tokenizer to use for tokenization.
        block_size (int, optional): The maximum length of the tokenized
        sequence. Defaults to None.
        add_eos (bool, optional): Whether to add an end-of-sequence token.
        Defaults to False.

    Returns:
        Dict: A dictionary containing tokenized input ids, input ids length,
        and labels length.
    """
    if block_size is not None:
        model_max_length = min(block_size, tokenizer.model_max_length)
    else:
        model_max_length = tokenizer.model_max_length
    tokenized = tokenizer(
        text,
        padding="longest",
        max_length=model_max_length,
        truncation=True,
        add_special_tokens=False,
    )
    if add_eos:
        tokenized.input_ids = tokenized.input_ids + [tokenizer.eos_token_id]
    if isinstance(tokenized.input_ids, torch.Tensor):
        input_ids = tokenized.input_ids
    else:
        input_ids = torch.Tensor(tokenized.input_ids)
    input_ids_lens = labels_lens = \
        input_ids.ne(tokenizer.pad_token_id).sum().item()
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def tokenized_dec_func(example, tokenizer: transformers.PreTrainedTokenizer,
                       block_size=None, add_eos=False) -> Dict:
    """Tokenize source and target texts separately for decoder input.

    Args:
        example (Dict): A dictionary containing 'source' and 'target' texts.
        tokenizer (transformers.PreTrainedTokenizer): The pre-trained
        tokenizer to use for tokenization.
        block_size (int, optional): The maximum length of the tokenized
        sequence. Defaults to None.
        add_eos (bool, optional): Whether to add an end-of-sequence token.
        Defaults to False.

    Returns:
        Dict: A dictionary containing tokenized input ids and label ids for
        decoder input.
    """
    source = example['source']
    target = example['target']
    source_target = source + ' ' + target  # Add a space
    example_tokenized = tokenize_fn(source_target, tokenizer,
                                    block_size, add_eos=add_eos)
    target_tokenized = tokenize_fn(target, tokenizer, block_size,
                                   add_eos=add_eos)
    input_ids = example_tokenized["input_ids"]
    labels = input_ids.clone()
    labels[:-len(target_tokenized["input_ids"])] = IGNORE_INDEX
    labels[input_ids.eq(tokenizer.pad_token_id)] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def tokenized_encdec_func(example, tokenizer: transformers.PreTrainedTokenizer,
                          block_size=None, add_eos=False) -> Dict:
    """Tokenize source and target texts separately for encoder-decoder input.

    Args:
        example (Dict): A dictionary containing 'source' and 'target' texts.
        tokenizer (transformers.PreTrainedTokenizer): The pre-trained
        tokenizer to use for tokenization.
        block_size (int, optional): The maximum length of the tokenized
        sequence. Defaults to None.
        add_eos (bool, optional): Whether to add an end-of-sequence token.
        Defaults to False.

    Returns:
        Dict: A dictionary containing tokenized input ids and label ids for
        encoder-decoder input.
    """
    source = example['source']
    target = example['target']
    source_tokenized = tokenize_fn(source, tokenizer,
                                   block_size, add_eos=False)
    target_tokenized = tokenize_fn(target, tokenizer,
                                   block_size, add_eos=add_eos)
    input_ids = source_tokenized["input_ids"]
    labels = target_tokenized["input_ids"]
    labels[labels.eq(tokenizer.pad_token_id)] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


IGNORE_INDEX = -100


def pad_first(input_ids: List[torch.Tensor], padding_value: int,
              max_length: int = None):
    """Pad input sequences to the beginning.

    Args:
        input_ids (List[torch.Tensor]): A list of input sequences represented
        as tensors.
        padding_value (int): The value to pad the sequences with.
        max_length (int, optional): The maximum length of the padded sequences.
        If None, it will be the maximum length of input sequences.
        Defaults to None.

    Returns:
        torch.Tensor: Padded sequences with the same length.
    """
    if max_length is None:
        max_len = max(len(x) for x in input_ids)
    else:
        max_len = max_length
    padded_input_ids = torch.full((len(input_ids), max_len),
                                  padding_value, dtype=torch.long)
    for i, x in enumerate(input_ids):
        padded_input_ids[i, -len(x):] = x
    return padded_input_ids


def pad_last(input_ids: List[torch.Tensor], padding_value: int,
             max_length: int = None):
    """Pad input sequences to the end.

    Args:
        input_ids (List[torch.Tensor]): A list of input sequences represented
        as tensors.
        padding_value (int): The value to pad the sequences with.
        max_length (int, optional): The maximum length of the padded sequences.
        If None, it will be the maximum length of input sequences.
        Defaults to None.

    Returns:
        torch.Tensor: Padded sequences with the same length.
    """
    if max_length is None:
        max_len = max(len(x) for x in input_ids)
    else:
        max_len = max_length
    padded_input_ids = torch.full((len(input_ids), max_len),
                                  padding_value, dtype=torch.long)
    for i, x in enumerate(input_ids):
        padded_input_ids[i, :len(x)] = x
    return padded_input_ids
