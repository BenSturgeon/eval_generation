
class ConstrainPromptsCreatedMeta(type):
    """We want n_prompts_created_per_generation to be a reasonable batch size."""
    def __new__(cls, name, bases, class_dict):
        original_init = class_dict.get('__init__')
        
        def new_init(self, *args, **kwargs):
            if original_init:
                original_init(self, *args, **kwargs)
            if 'n_prompts_created_per_generation' in kwargs and \
                    kwargs['n_prompts_created_per_generation'] not in [1, 2, 3, 5, 10, 20]:
                raise ValueError("n_prompts_created_per_generation must be one of: 1, 2, 3, 5, 10, 20")
            
        class_dict['__init__'] = new_init

        return type.__new__(cls, name, bases, class_dict)


class PromptBase(metaclass=ConstrainPromptsCreatedMeta):
    def __init__(self, entropy_file: str = 'cases/adding_entropy.txt') -> None:
        with open(entropy_file, 'r') as f:
            self.entropy_lines = f.readlines()

    def generative_prompt(self) -> str:
        raise NotImplementedError
    
    def generative_system_prompt(self) -> str:
        return ""
    
    def relevance_check_prompt(self, question: str) -> str:
        raise NotImplementedError
    
    def relevance_check_system_prompt(self) -> str:
        return ""
    
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        raise NotImplementedError
    
    def subject_evaluation_system_prompt(self) -> str:
        return ""
    
    # def get_optional_subject_prompt_prepend(self) -> str:
    #     return ""
    
    def get_top_eval_score(self) -> int:
        raise NotImplementedError

    def extract_score(self, evaluator_response: str) -> float:
        """Extract numerical score from evaluator response. Override for custom parsing.
        
        Args:
            evaluator_response: The raw response from the evaluator model
            
        Returns:
            float: The extracted score
            
        Raises:
            NotImplementedError: If no custom parsing is implemented and default XML parsing fails
        """
        from src.utils import extract_score_from_xml
        try:
            return float(extract_score_from_xml(evaluator_response))
        except (ValueError, AttributeError):
            raise NotImplementedError(
                f"Default XML parsing failed. Override extract_score() for custom response format.\n"
                f"Response: {evaluator_response[:200]}..."
            )
    
    def get_score_range(self) -> tuple[float, float]:
        """Return (min_score, max_score) for this evaluation type.
        
        Returns:
            tuple: (minimum_possible_score, maximum_possible_score)
        """
        return (0.0, float(self.get_top_eval_score()))

    def subject_model_system_prompt(self) -> str:
        return ""
