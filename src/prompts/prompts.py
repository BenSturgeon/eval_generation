from typing import Dict
from src.prompts.prompt_base import PromptBase
from src.prompts.generic_template import GenericPrompt
from src.prompts.long_term_goal_triggers import GoalConflictPrompt


prompt_objects: Dict[str, PromptBase] = {
    "generic": GenericPrompt,
    "goal_conflict": GoalConflictPrompt,
}