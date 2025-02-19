import re
from typing import Dict, List

from m2d.config import FormatConfig


class Formatter:
    """
        Since our model can output response in different format
        We use the formatter to process input and output
    """
    def __init__(self, format_config: FormatConfig):
        self.format_config = format_config

    @classmethod
    def from_path(cls, path: str):
        return cls(format_config=FormatConfig.from_path(path))

    def format_input_str(
        self, 
        task: str,
        input_str: str, 
    ) -> str:
        task_config = self.format_config.task_configs.get(task, None)
        if task_config:
            input_str = \
                task_config.get("prefix_inst", "") + \
                input_str + \
                task_config.get("suffix_inst", "")
        return input_str
    
    def format_input_conversation(
        self, 
        task: str,
        conversation: List[Dict], 
    ) -> str:
        task_config = self.format_config.task_configs.get(task, None)
        # only applies on the last user input
        if task_config:
            for turn_idx in reversed(range(len(conversation))):
                if conversation[turn_idx]["role"] == "user":
                    conversation[turn_idx]["content"] = \
                        task_config.get("prefix_inst", "") + \
                        conversation[turn_idx]["content"] + \
                        task_config.get("suffix_inst", "")
                    break
        return conversation

    def parse_output(
        self, 
        task: str,
        output_str: str, 
    ) -> str:
        task_config = self.format_config.task_configs.get(task, None)
        if task_config and "parse_regex" in task_config:
            match = re.search(task_config["parse_regex"], output_str)
            if match:
                output_str = match.group()

        return output_str
