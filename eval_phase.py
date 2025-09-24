from pipeline import *
from src.evaluate_model_existing import evaluate_many_subject_models_existing
from src.batching import install_batching_config

def eval_phase(evaluations_config_file, output_folder = "output/eval_output", input_folder = "output/generation_output", use_existing_responses=False):
    setup_keys(KEYS_PATH)
    config = load_config(evaluations_config_file)
    install_batching_config(config)

    problem_types = config['general_params']['problem_types']
    del config['general_params']['problem_types']
    if not problem_types:
        raise ValueError("No problem types specified in config. Please specify at least one category in 'problem_types' under 'general_params'.")

    for problem_type in problem_types:

        print("Running evaluation for problem type:", problem_type)

        if not os.path.exists(os.path.join(input_folder, problem_type, "generated.csv")):
            raise FileNotFoundError(f"Input folder for {problem_type} does not exist. Please run the generation phase first.")

        prompt_object = prompt_objects[problem_type]()
        results_output_folder = os.path.join(output_folder, problem_type)
        os.makedirs(os.path.join(results_output_folder), exist_ok=True)

        with open(os.path.join(input_folder, problem_type, "generated.csv"), 'r') as f:
            is_diverse_df = pd.read_csv(f)

        model_evaluation_html, model_scores_html, out_df = evaluate_and_visualize_model(is_diverse_df, config, prompt_object)
        html_out = "<h1>Model evaluation phase</h1>"
        html_out += model_scores_html
        html_out += model_evaluation_html

        # Save as parquet with proper type conversion
        out_df_parquet = out_df.copy()
        for col in out_df_parquet.columns:
            if out_df_parquet[col].dtype == 'object':
                out_df_parquet[col] = out_df_parquet[col].apply(lambda x: str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x)
        out_df_parquet.to_parquet(os.path.join(results_output_folder, 'raw.parquet'), index=False)
        # Also save as CSV for backwards compatibility
        with open(os.path.join(results_output_folder, 'raw.csv'), 'w') as f:
            f.write(out_df.to_csv(index=False))
        
        with open(os.path.join(results_output_folder, 'plot.html'), 'w') as f:
            f.write(html_out)

if __name__ == "__main__":
    argh.dispatch_command(eval_phase)