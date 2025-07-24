from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

import pandas as pd
import google.generativeai.types as generation_types

from src.prompts.prompt_base import PromptBase
from src.llms import LLM
from src.utils import hash_cache


N_CONCURRENT_REQUESTS = 1000


@hash_cache()
def model_response(prompt, system_prompt, model, temperature, top_p, max_tokens):
    # Removed semaphore acquisition logic for provider-specific throttling
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
            prompts=list(prompts_tuple),  # Convert back to list
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stage_name=stage_name,
        )

    # --- Fast path: Use Batch API if possible and all system prompts are the same ---
    if use_batching and model.startswith(("claude", "gpt-", "o")) and len(set(system_prompts)) == 1:
        try:
            # Pass a tuple of prompts to make it hashable for the cache
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

    # --- Fallback: Original multi-threaded per-prompt logic ---
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
            # model_response returns a tuple (response, system_prompt), we just need the response
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
):
    """
    Computes scores for a list of subject model responses using a specified evaluator model.
    """
    # 1. Prepare the prompts for the evaluator model
    eval_prompts = []
    eval_system_prompts = []
    system_prompt = prompt_object.subject_evaluation_system_prompt()
    for i, (prompt, response) in enumerate(zip(prompts, subject_responses)):
        eval_prompts.append(
            prompt_object.subject_evaluation_prompt(prompt, response)
        )
        eval_system_prompts.append(system_prompt)

    # 2. Get responses from the evaluator model
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
        use_batching=True, # Always attempt to batch for the judge
    )

    # 3. Calculate scores from the evaluator's responses
    scores = []
    error_placeholders = [
        "[RATE LIMIT ERROR DURING LLM CHAT]",
        "[ERROR DURING LLM CHAT]",
        "[GENERATION STOPPED DUE TO RECITATION]",
        "[ANTHROPIC OVERLOAD ERROR DURING LLM CHAT]"
    ]
    for evaluator_response in evaluator_responses:
        if evaluator_response in error_placeholders:
            # Use minimum score for error cases
            min_score, _ = prompt_object.get_score_range()
            score = min_score
        else:
            try:
                score = prompt_object.extract_score(evaluator_response)
            except (ValueError, NotImplementedError) as e:
                print(f"Error extracting score from evaluator '{evaluator_model}'. Using minimum score. Error: {e}")
                min_score, _ = prompt_object.get_score_range()
                score = min_score

        scores.append(score)

    return scores, eval_system_prompts, eval_prompts, evaluator_responses


def evaluate_model(
    prompts,
    evaluator_model,
    subject_model,
    subject_model_temperature,
    subject_model_top_p, subject_max_tokens, prompt_object,
    use_cache, refresh_cache,
    evaluator_max_tokens: int = 5000,
    gemini_max_tokens: int = 8192,
    use_batching_for_subjects: bool = False,
): # Note: gemini_max_tokens default isn't used if passed from config

    subject_model_system_prompt = [prompt_object.subject_model_system_prompt() for _ in range(len(prompts))]

    # Select the appropriate max_tokens based on the specific subject model
    specific_gemini_models = [
        "models/gemini-2.5-pro-preview-03-25",
        "models/gemini-2.5-flash-preview-04-17"
    ]
    
    # Apply gemini_max_tokens if the model is one of the specific Gemini models
    if subject_model in specific_gemini_models:
        current_subject_max_tokens = gemini_max_tokens
        print(f"Using gemini_max_tokens ({gemini_max_tokens}) for {subject_model}") # Optional: logging
    else:
        current_subject_max_tokens = subject_max_tokens

    subject_responses, subject_system_prompts = get_model_responses(
        prompts=prompts, 
        system_prompts=subject_model_system_prompt, 
        model=subject_model, 
        temperature=subject_model_temperature, 
        top_p=subject_model_top_p,
        max_tokens=current_subject_max_tokens, 
        use_cache=use_cache, 
        refresh_cache=refresh_cache,
        use_batching=use_batching_for_subjects,
    )
    
    scores, evaluator_system_prompts, evaluator_prompts, evaluator_responses = \
        get_scores(
            prompts=prompts,
            subject_responses=subject_responses,
            prompt_object=prompt_object,
            evaluator_model=evaluator_model, 
            use_cache=use_cache, 
            refresh_cache=refresh_cache, 
            subject_model=subject_model, 
            evaluator_max_tokens=evaluator_max_tokens
        )
    
    return scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses


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
    gemini_max_tokens: int = 8192,
    evaluator_max_tokens: int = 8192,
    use_batching_for_subjects: bool = False,
) -> pd.DataFrame:
    dfs = []

    for subject_model in tqdm(subject_models, desc="Evaluating subject models"):
        scores, subject_responses, subject_system_prompts, evaluator_system_prompts, evaluator_prompts, evaluator_responses = evaluate_model(
            prompts=prompts,
            evaluator_model=evaluator_model,
            subject_model=subject_model,
            subject_model_temperature=subject_model_temperature,
            subject_model_top_p=subject_model_top_p,
            subject_max_tokens=subject_max_tokens,
            prompt_object=prompt_object,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            gemini_max_tokens=gemini_max_tokens,
            evaluator_max_tokens=evaluator_max_tokens,
            use_batching_for_subjects=use_batching_for_subjects,
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

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df