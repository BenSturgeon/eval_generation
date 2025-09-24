from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import pandas as pd
import google.generativeai.types as generation_types

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache, sum_deductions_from_json


N_CONCURRENT_REQUESTS = 50


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens):
    response = ""
    llm = LLM(model, system_prompt)
    try:
        response = llm.chat(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    except generation_types.StopCandidateException as e:
        print(f"Warning: Gemini model '{model}' stopped due to RECITATION. Prompt: '{prompt[:100]}...' Exception: {e}")
        response = "[GENERATION STOPPED DUE TO RECITATION]"
    except Exception as e:
        print(f"Error during llm.chat for model {model}: {e}")
        status_code = getattr(getattr(e, 'response', None), 'status_code', None)
        if status_code == 429:
            response = "[RATE LIMIT ERROR DURING LLM CHAT]"
        elif status_code == 529:
            response = "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
        else:
            response = "[ERROR DURING LLM CHAT]"
    return response, system_prompt


def get_model_responses(
    prompts,
    system_prompts,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_cache: bool,
    refresh_cache: bool,
    stage_name: str = "Subject Model Response",
    use_batching: bool = False,
):
    """
    Retrieve model responses for a list of prompts, handling batching and caching.
    """
    
    # --- Caching for Batchable Models ---
    @hash_cache()
    def _cached_batch_call(prompts_tuple, system_prompt, model, max_tokens, temperature, top_p, stage_name):
        from src.batch_utils import batch_model_response
        return batch_model_response(
            prompts=list(prompts_tuple),
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stage_name=stage_name,
        )

    # Fast path: Use Batch API if possible and all system prompts are the same
    # OpenRouter models (like "openai/o3") should not use OpenAI batch API
    is_openrouter = "/" in model
    if use_batching and not is_openrouter and model.startswith(("claude", "gpt-", "o")) and len(set(system_prompts)) == 1:
        print(f"[DEBUG] Using batch API path for {model} with {len(prompts)} prompts")
        try:
            responses = _cached_batch_call(
                prompts_tuple=tuple(prompts),
                system_prompt=system_prompts[0],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stage_name=stage_name,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            )
            return responses, system_prompts
        except Exception as e:
            print(f"[WARN] Batch path failed for model {model}: {e}. Falling back to per-prompt mode.")

    # Fallback: Original multi-threaded per-prompt logic
    responses = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=N_CONCURRENT_REQUESTS) as executor:
        future_to_index = {
            executor.submit(
                model_response,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_cache=use_cache,
                refresh_cache=refresh_cache,
            ): i
            for i, (prompt, system_prompt) in enumerate(zip(prompts, system_prompts))
        }
        for future in tqdm(as_completed(future_to_index), total=len(prompts), desc=f"Getting responses for {stage_name}: {model}"):
            index = future_to_index[future]
            responses[index], _ = future.result()
            
    return responses, system_prompts


def get_scores(
    prompts,
    subject_responses,
    prompt_object,
    evaluator_model,
    use_cache,
    refresh_cache,
    subject_model,
    evaluator_max_tokens: int = 5000,
    misinformation: List[str] = None,
):
    """
    Computes scores for a list of subject model responses using a specified evaluator model.
    """
    eval_prompts = []
    eval_system_prompts = []
    system_prompt = prompt_object.subject_evaluation_system_prompt()
    for i, (prompt, response) in enumerate(zip(prompts, subject_responses)):
        response_text = response.get('content', response) if isinstance(response, dict) else response
        
        if misinformation:
            eval_prompts.append(
                prompt_object.subject_evaluation_prompt(prompt, response_text, misinformation[i])
            )
        else:
            eval_prompts.append(
                prompt_object.subject_evaluation_prompt(prompt, response_text)
            )
        eval_system_prompts.append(system_prompt)

    print(f"[DEBUG] Calling evaluator model {evaluator_model} with {len(eval_prompts)} evaluation prompts...")
    evaluator_responses, _ = get_model_responses(
        prompts=eval_prompts,
        system_prompts=eval_system_prompts,
        model=evaluator_model,
        temperature=0,
        top_p=1,
        max_tokens=evaluator_max_tokens,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        stage_name="Evaluator Scoring",
        use_batching=True,
    )
    print(f"[DEBUG] Got {len(evaluator_responses)} evaluator responses")

    scores = []
    error_placeholders = [
        "[RATE LIMIT ERROR DURING LLM CHAT]",
        "[ERROR DURING LLM CHAT]",
        "[GENERATION STOPPED DUE TO RECITATION]",
        "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
    ]
    for evaluator_response in evaluator_responses:
        total_deduction = 0
        if isinstance(evaluator_response, dict):
            response_text = evaluator_response.get('content', str(evaluator_response))
        else:
            response_text = evaluator_response
            
        if response_text in error_placeholders:
            total_deduction = prompt_object.top_eval_score
        else:
            try:
                total_deduction = sum_deductions_from_json(response_text, prompt_object.deductions)
            except ValueError as e:
                print(f"Error parsing JSON from evaluator '{evaluator_model}'. Assigning max deduction. Error: {e}")
                print(f"Original response: {evaluator_response}")
                total_deduction = prompt_object.top_eval_score

        score = max(prompt_object.top_eval_score - total_deduction, 0)
        scores.append(score)

    return scores, eval_system_prompts, eval_prompts, evaluator_responses


def evaluate_model(
    prompts,
    evaluator_model,
    subject_model,
    subject_responses,  # Pre-existing responses
    subject_system_prompts,  # Pre-existing system prompts
    prompt_object,
    use_cache,
    refresh_cache,
    evaluator_max_tokens: int = 5000,
    misinformation: List[str] = None,
):
    """Evaluate existing subject model responses with a new evaluator."""
    
    print(f"[DEBUG] Using existing {len(subject_responses)} subject responses for {subject_model}")
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(
            prompts=prompts,
            subject_responses=subject_responses,
            prompt_object=prompt_object,
            evaluator_model=evaluator_model, 
            use_cache=use_cache, 
            refresh_cache=refresh_cache, 
            subject_model=subject_model, 
            evaluator_max_tokens=evaluator_max_tokens,
            misinformation=misinformation
        )
    
    return scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses


def evaluate_model_new_subject(
    prompts,
    evaluator_model,
    subject_model,
    subject_model_temperature,
    subject_model_top_p, subject_max_tokens, prompt_object,
    use_cache, refresh_cache,
    evaluator_max_tokens: int = 5000,
    use_batching_for_subjects: bool = False,
    misinformation: List[str] = None,
):

    subject_model_system_prompt = [prompt_object.subject_model_system_prompt() for _ in range(len(prompts))]

    subject_responses, subject_system_prompts = get_model_responses(
        prompts=prompts, 
        system_prompts=subject_model_system_prompt, 
        model=subject_model, 
        temperature=subject_model_temperature, 
        top_p=subject_model_top_p,
        max_tokens=subject_max_tokens, 
        use_cache=use_cache, 
        refresh_cache=refresh_cache,
        use_batching=use_batching_for_subjects,
    )
    
    error_placeholders = ["[RATE LIMIT ERROR DURING LLM CHAT]", "[ERROR DURING LLM CHAT]", "[GENERATION STOPPED DUE TO RECITATION]", "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"]
    error_count = sum(1 for r in subject_responses if r in error_placeholders)
    print(f"[DEBUG] Subject responses: {len(subject_responses) - error_count} successful, {error_count} errors")
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(
            prompts=prompts,
            subject_responses=subject_responses,
            prompt_object=prompt_object,
            evaluator_model=evaluator_model, 
            use_cache=use_cache, 
            refresh_cache=refresh_cache, 
            subject_model=subject_model, 
            evaluator_max_tokens=evaluator_max_tokens,
            misinformation=misinformation
        )
    
    return scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses


def evaluate_many_subject_models_existing(
    existing_df: pd.DataFrame,
    evaluator_model: str,
    prompt_object: PromptBase,
    use_cache: bool,
    refresh_cache: bool,
    evaluator_max_tokens: int = 8192,
    max_concurrent_subjects: int = 1,
) -> pd.DataFrame:
    """Evaluate existing subject model responses with a new evaluator."""
    dfs = []
    
    subject_models = existing_df['subject_model'].unique()
    
    for subject_model in tqdm(subject_models, desc="Re-evaluating subject models"):
        model_df = existing_df[existing_df['subject_model'] == subject_model].copy()
        
        prompts = model_df['prompt'].tolist()
        subject_responses = model_df['subject_response'].tolist()
        subject_system_prompts = model_df['subject_system_prompt'].tolist()
        
        misinformation = None
        if 'misinformation' in model_df.columns:
            misinformation = model_df['misinformation'].tolist()
        
        scores, _, _, evaluator_system_prompts, evaluator_prompts, evaluator_responses = evaluate_model(
            prompts=prompts,
            evaluator_model=evaluator_model,
            subject_model=subject_model,
            subject_responses=subject_responses,
            subject_system_prompts=subject_system_prompts,
            prompt_object=prompt_object,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            evaluator_max_tokens=evaluator_max_tokens,
            misinformation=misinformation
        )
        
        df = pd.DataFrame({
            'prompt': prompts,
            'score': scores,
            'subject_response': subject_responses,
            'subject_system_prompt': subject_system_prompts,
            'evaluator_system_prompt': evaluator_system_prompts,
            'evaluator_prompt': evaluator_prompts,
            'evaluator_response': evaluator_responses,
            'subject_model': subject_model
        })
        
        # Add misinformation column if it exists
        if misinformation:
            df['misinformation'] = misinformation
        
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    return df


def evaluate_many_subject_models(
    prompts: List[str],
    subject_models: List[str],
    evaluator_model: str,
    subject_model_temperature: float,
    subject_model_top_p: float,
    subject_max_tokens: int,
    prompt_object: PromptBase,
    use_cache: bool,
    refresh_cache: bool,
    misinformation: List[str] = None,
    evaluator_max_tokens: int = 8192,
    use_batching_for_subjects: bool = False,
    max_concurrent_subjects: int = 1,
) -> pd.DataFrame:
    dfs = []

    for subject_model in tqdm(subject_models, desc="Evaluating subject models"):
        scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses = evaluate_model_new_subject(
            prompts=prompts,
            evaluator_model=evaluator_model,
            subject_model=subject_model,
            subject_model_temperature=subject_model_temperature,
            subject_model_top_p=subject_model_top_p,
            subject_max_tokens=subject_max_tokens,
            prompt_object=prompt_object,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            evaluator_max_tokens=evaluator_max_tokens,
            use_batching_for_subjects=use_batching_for_subjects,
            misinformation=misinformation,
        )

        df = pd.DataFrame({
            'prompt': prompts,
            'score': scores,
            'subject_response': subject_responses,
            'subject_system_prompt': subject_system_prompts,
            'evaluator_system_prompt': evaluator_system_prompts,
            'evaluator_prompt': evaluator_prompts,
            'evaluator_response': evaluator_responses,
            'subject_model': subject_model
        })

        # Add misinformation column if it exists
        if misinformation:
            df['misinformation'] = misinformation

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df