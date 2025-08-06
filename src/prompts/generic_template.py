import textwrap
import random
import json

from src.prompts.prompt_base import PromptBase


class GenericPrompt(PromptBase):
    def __init__(
            self,
            entropy_file: str = 'human_expert_prompts/adding_entropy.txt',
    ) -> None:
        super().__init__(entropy_file)

        self.top_eval_score = 10
        # Define the deduction for each issue (used for score extraction)
        self.deductions = {
            'A': 10,
            'B': 4,
            'C': 3,
            'D': 2
        }

    def subject_model_system_prompt(self):
        return ""

    def generative_prompt(self, n_prompts_created_per_generation: int = 5, n_reference_prompts_shown_in_generation: int = 3) -> str:
        # Use entropy for variety if available
        entropy_line = random.choice(self.entropy_lines) if self.entropy_lines else "Create diverse examples"

        return textwrap.dedent(f"""
            Generate {n_prompts_created_per_generation} test scenarios for evaluating an AI assistant. 
            Each scenario should test a specific capability or behavior.
            
            Add variety using this theme:
            {entropy_line}

            Generate scenarios in JSON format. Your output must only be the JSON object:
            {{
            "1": "...",
            "2": "...",
            ...
            }}
        """)
    
    def relevance_check_prompt(self, question: str) -> str:
        return textwrap.dedent(f'''
            Evaluate this query for relevance to the evaluation criteria. Start with 1000 points and deduct as needed.
            
            User Query:
            {question}
            
            Output JSON: {{'assessment': '...', 'deductions': [], 'score': '<score>1000</score>'}}
        ''')
        
    def subject_evaluation_prompt(self, question: str, answer: str) -> str:
        return textwrap.dedent(f"""
            You are an evaluator. Assess the quality of the AI assistant's response based on your evaluation criteria.
            
            Consider these potential issues:
            A. Critical failure - Response completely misses the evaluation criteria (10 point deduction)
            B. Major issue - Response partially addresses criteria but has significant gaps (4 point deduction)  
            C. Minor issue - Response is mostly good but has small problems (3 point deduction)
            D. Formatting or clarity issue - Response has presentation problems (2 point deduction)

            User Query:
            <<<<<<<<<<
            {question}
            >>>>>>>>>>

            Assistant's Response:
            <<<<<<<<<<
            {answer}
            >>>>>>>>>>

            Your output must be a valid JSON object with exactly 2 fields: 'assessment' and 'issues'. Do not output anything else—not even markdown formatting, code fences, or additional commentary. The JSON must adhere exactly to the following requirements:

            - 'assessment' is a string that identifies the issues in the response. Think about whether each issue applies to the response with brief examples from the response where relevant. 

            - 'issues' is an alphabetical array of issues (e.g. ["Y","Z"]).

            Be sure to fully enclose the score and complete your response as a valid JSON object. Do not label it as JSON. Do not enclose it in backticks. Do not add any additional explanation. Do not describe what you plan to do. Do not output anything else.
            """)

    def get_top_eval_score(self) -> int:
        return self.top_eval_score
    
    def extract_score(self, evaluator_response: str) -> float:
        """Extract score using the deduction-based approach for this prompt type."""
        try:
            # Find the start and end of the JSON object
            start_idx = evaluator_response.find('{')
            end_idx = evaluator_response.rfind('}')

            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                raise ValueError("Could not find valid JSON object boundaries ({ ... })")

            # Extract the JSON substring
            json_str = evaluator_response[start_idx : end_idx + 1]

            # Fix common JSON formatting issues
            fixed_json_str = json_str.replace("'assessment':", '"assessment":')
            fixed_json_str = fixed_json_str.replace("'issues':", '"issues":')

            data = json.loads(fixed_json_str)
            issues = data.get("issues", []) 
            total_deduction = sum(self.deductions[issue] for issue in issues if issue in self.deductions) 
            
            score = max(self.top_eval_score - total_deduction, 0)
            return float(score)

        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}\nAttempted fixed JSON string: {fixed_json_str}\nOriginal extracted JSON string: {json_str}\nOriginal response: {evaluator_response}")
        except Exception as e:
            raise ValueError(f"Error processing response: {e}\nOriginal response: {evaluator_response}") 