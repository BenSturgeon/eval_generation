from typing import List, Union
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from tqdm import tqdm

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache
from src.prompts.prompts import prompt_objects

N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def generate_single_prompt(
    model: str, 
    generative_prompt: str,
    system_prompt: str,
    n_prompts_created_per_generation: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_retries: int = 10
):

    for _ in range(max_retries):  
        llm = LLM(model, system_prompt=system_prompt)

        response = llm.chat(prompt=generative_prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, return_json=True)

        if "```json" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        elif "```" in response:
            response = response.replace("```", "").strip()

        try:
            prompt = json.loads(response)
            
            if isinstance(prompt, dict):
                if "data" in prompt and isinstance(prompt["data"], list):
                    prompt = prompt["data"]
                elif "response" in prompt and isinstance(prompt["response"], list):
                    prompt = prompt["response"]
                else:
                    generated_subject_prompts = [json.dumps(prompt)]
                    if len(generated_subject_prompts) == n_prompts_created_per_generation:
                        return generated_subject_prompts, system_prompt, generative_prompt
                    else:
                        continue
            
            if isinstance(prompt, list):
                if all(isinstance(p, dict) for p in prompt):
                    generated_subject_prompts = [json.dumps(p) for p in prompt]
                else:
                    generated_subject_prompts = prompt
            else:
                continue
        except json.JSONDecodeError as e:
            continue

        if len(generated_subject_prompts) != n_prompts_created_per_generation:
            continue

        return generated_subject_prompts, system_prompt, generative_prompt
    
    raise Exception(f"Failed to generate prompts after {max_retries} retries: \nLast response: {response}")


def generate_dataset(
    model: str,
    n_prompts: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    prompt_object: PromptBase,
    n_prompts_created_per_generation: int,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Union[List[str], List[str], List[str]]:

    requests_needed = (n_prompts + n_prompts_created_per_generation - 1) // n_prompts_created_per_generation
    generated_prompts = [None] * n_prompts

    random.seed(42)
    system_prompts = [prompt_object.generative_system_prompt() for _ in range(n_prompts)]
    generative_prompts = [prompt_object.generative_prompt(n_prompts_created_per_generation) for _ in range(n_prompts)]

    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {executor.submit(
            generate_single_prompt,
            model=model,
            generative_prompt=generative_prompts[i],
            system_prompt=system_prompts[i],
            n_prompts_created_per_generation=n_prompts_created_per_generation,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            cache_nonce=i,
            use_cache=use_cache,
            refresh_cache=refresh_cache
            ): i for i in range(requests_needed)
        }
        for future in tqdm(as_completed(future_to_index), total=requests_needed, desc='Generating prompts'):

            generated_prompt, system_prompt, generative_prompt = future.result()
            index = future_to_index[future]
            start_idx = index * n_prompts_created_per_generation
            end_idx = (index + 1) * n_prompts_created_per_generation

            index = future_to_index[future]
            generated_prompts[start_idx: end_idx] = generated_prompt
            system_prompts[start_idx: end_idx] = [system_prompt] * n_prompts_created_per_generation
            generative_prompts[start_idx: end_idx] = [generative_prompt] * n_prompts_created_per_generation

    return generated_prompts, system_prompts, generative_prompts
