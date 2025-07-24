# Generation Self-Steering Instructions

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   # Using uv (recommended for speed):
   uv pip install -r requirements.txt
   
   # Or using pip:
   pip install -r requirements.txt
   ```

3. Set up API keys:
   Create a `keys.json` file in the root directory with your API keys:
   ```json
   {
     "OPENAI_API_KEY": "your-openai-api-key",
     "ANTHROPIC_API_KEY": "your-anthropic-api-key",
     "GROQ_API_KEY": "your-groq-api-key",
     "GOOGLE_API_KEY": "your-google-api-key",
     "GROK_API_KEY": "your-grok-api-key",
     "DEEPSEEK_API_KEY": "your-deepseek-api-key",
     "REPLICATE_API_TOKEN": "your-replicate-token"
   }
   ```
   Note: You only need to include keys for the providers you plan to use.

## Configuration

Edit `configs/example_config.yaml` to configure your generation:

### Key Parameters:

- **use_cache**: Set to `True` to cache API responses (saves money on repeated runs), `False` to always make fresh API calls
- **refresh_cache**: Set to `True` to ignore existing cache and regenerate all responses
- **problem_types**: List of prompt types to generate (e.g., `goal_conflict`, `generic`)
- **model**: The model to use for generation (e.g., `gpt-4o-mini`, `claude-3-opus`, etc.)
- **n_prompts**: Total number of prompts to generate
- **n_prompts_created_per_generation**: Number of prompts per API call (must be 1, 2, 3, 5, 10, or 20)

### Cache Management

The system uses a file-based cache in the `hash_cache/` directory:
- Enable caching with `use_cache: True` to save API costs during development
- Clear cache by deleting the `hash_cache/` directory or setting `refresh_cache: True`
- Cache is based on exact prompt + parameters, so any change will trigger a new API call

## Running the Pipeline

### Generation Phase
Generate prompts based on your configuration:
```bash
python3 generation_phase.py configs/example_config.yaml
```

### Full Pipeline
Run generation, quality evaluation, diversity analysis, and model evaluation:
```bash
python3 pipeline.py configs/example_config.yaml
```

### Evaluation Only
Evaluate existing prompts against subject models:
```bash
python3 eval_phase.py configs/example_config.yaml
```

## Adding New Prompt Types

1. Create a new prompt class in `src/prompts/` that inherits from `PromptBase`
2. Implement required methods:
   - `generative_prompt()`: Prompt for generating examples
   - `relevance_check_prompt()`: Prompt for quality evaluation
   - `subject_evaluation_prompt()`: Prompt for evaluating model responses
   - `get_top_eval_score()`: Maximum score for evaluation
   - `extract_score()`: (Optional) Custom score extraction logic

3. Register your prompt class in `src/prompts/prompts.py`:
   ```python
   prompt_objects = {
       "your_prompt_type": YourPromptClass,
       ...
   }
   ```

4. Add your prompt type to the config file under `problem_types`

## Cost Tracking

The system automatically tracks API costs:
- Summary: `cost_logs/cost_summary.json`
- Detailed log: `cost_logs/cost_detailed_log.jsonl`
- Costs are tracked per model and per run

## Troubleshooting

### "Client.__init__() got an unexpected keyword argument 'proxies'" Error
This occurs due to incompatibility between OpenAI and httpx versions. The requirements.txt pins compatible versions:
- openai==1.12.0
- httpx==0.25.2

### Invalid Score Format Errors
If you're implementing a custom prompt type, ensure your evaluation prompt generates responses in the expected format. Override the `extract_score()` method if using a non-XML format.

### API Key Errors
Ensure your `keys.json` file exists and contains valid API keys for the models you're using.

## Output Files

Generated files are saved in:
- `generated_data/<problem_type>/`: Raw generated prompts
- `scored_data/<problem_type>/`: Prompts with quality scores
- `diversity_data/<problem_type>/`: Diversity analysis results
- `eval_data/<problem_type>/`: Model evaluation results
- `visualizations/<problem_type>/`: HTML visualizations