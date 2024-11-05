from typing import List, Tuple
import itertools

config_medfound = {
    "description": "Template used by medfound.",
    "user": "### User: ",
    "sep": "\n\n",
    "assistant": "### Assistant: ",
    "sep_input": "[[SEP]]",
}
config_plain = {
    "description": "Template used by default.",
    "user": "",
    "sep": "\n\n",
    "assistant": "",
    "sep_input": "[[SEP]]",
}
config_mapping = {
    "medfound": config_medfound,
    "plain": config_plain,
}


class SftPrompter(object):
    """Generate prompts for conversation based on specified template.

    Args:
        template_name (str, optional): Name of the template to use.
        Defaults to None.
        verbose (bool, optional): Flag to enable verbose mode.
        Defaults to False.
    """
    __slots__ = ("config", "_verbose")

    def __init__(self, template_name: str = None, verbose: bool = False):
        self._verbose = verbose
        if template_name is None:
            template_name = "medfound"

        self.config = config_mapping[template_name]
        self.config["prefixes"] = \
            [self.config["user"], self.config["assistant"]]

        if self._verbose:
            print("Using prompt template "
                  f"{template_name}: {self.config['description']}")

    def generate_prompt_response(self, texts: List[str],
                                 with_last=True) -> Tuple[str, str]:
        """Generate text for multiple rounds of dialogue

        Args:
            texts (List[str]): whether prompt contains the last segment of text
            with_last (bool, optional): Flag indicating whether to include the
            last segment of text in the prompt. Defaults to True,
                where
                - True: prompt contains the last segment of text for training
                reward models
                - False: prompt does not contain the last segment of text for
                training generation models or generation

        Returns:
            Tuple[str, str]: Tuple containing the generated prompt and the
            response.
        """
        if isinstance(texts, str):
            texts = texts.split(self.config["sep_input"])
        # when len(texts) is odd, add an empty string for inference
        if len(texts) % 2 == 1:
            texts = texts + [""]
        response = texts[-1]
        # If the last segment of text is not included, then mask the last
        # segment of text.
        if not with_last:
            texts = texts[:-1] + [""]
        texts = [prefix + text for text, prefix in zip(
            texts, itertools.cycle(self.config["prefixes"]))]
        prompt = self.config["sep"].join(texts)
        return prompt, response

    def add_prefix(self, texts) -> List[str]:
        texts_prompted = [prefix + text for text, prefix in zip(
            texts, itertools.cycle(self.config["prefixes"]))]
        return texts_prompted

    def get_response(self, output: str) -> str:
        return output.split(self.config["assistant"])[-1]
