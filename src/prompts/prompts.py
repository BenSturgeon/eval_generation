from typing import Dict
from src.prompts.prompt_base import PromptBase
from src.prompts.generic_template import GenericPrompt


prompt_objects: Dict[str, PromptBase] = {
    "generic": GenericPrompt,
}