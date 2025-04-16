# Synth_data
Framework for generating synthetic dialogue datasets using LLMs.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Create configuration file in `configs/generation_config.json`:
```json
{
    "generation_config": {
        "model": "google/gemma-2-27b-it",  // LLM to use
        "temperature": 0.8,
        "max_tokens": 4096,
        "total_input_tokens": 2048,
        "top_p": 0.95,
        "request_timeout": 500
    },
    "api_config": {
        "vllm_url": "http://0.0.0.0:8000/v1/chat/completions",
        "headers": {"Content-Type": "application/json"}
    },
    "dataset_config": {
        "source_dataset": "username/dataset",  // HuggingFace dataset to use as source
        "num_conversations": 5000,  // Number of conversations to generate
        "username": "your-username",  // Your HuggingFace username
        "dataset_name": "your-dataset-name",  // Name for the generated dataset
        "push_to_hub": true,  // Whether to upload to HuggingFace
        "private": true  // Whether the dataset should be private
    },
    "files": {
        "prompt_file": "prompts/dialogue_prompt.txt",  // Path to prompt template
        "base_dir": "experiments"  // Local directory for outputs
    }
}
```

3. Create prompt template in `prompts/dialogue_prompt.txt`:
```text
Your prompt template here with a {context} placeholder where 
the source conversation will be inserted.
```

## Usage

### Basic Usage
```bash
python generate_dialogues.py
```
This will use the default config at `configs/generation_config.json`

### Specify Config
```bash
python generate_dialogues.py --config configs/my_config.json
```

## Output Structure

The script creates:

1. **Local Output** (`experiments/dataset_name/`):
```
experiments/dataset_name/
├── configs/
│   └── generation_config.json
├── prompts/
│   └── dialogue_prompt.txt
└── [dataset files]
```

2. **HuggingFace Dataset** (if push_to_hub=true):
```
username/dataset_name/
├── configs/
│   └── generation_config.json
├── prompts/
│   └── dialogue_prompt.txt
└── [dataset files]
```

## Features

- **Checkpointing**: Saves progress every 100 conversations and enables resuming interrupted runs
- **Token Management**: Automatically handles context length within model limits
- **Validation**: Checks conversation format, content, and required fields
- **Analysis**: Generates statistics for each conversation including:
  - Message counts
  - Text lengths
  - Token usage
  - Context utilization
- **Backup**: Maintains copies of all configuration and prompt files

## Dataset Format

Each generated conversation contains:
1. Original Messages:
   - post_number
   - poster_id
   - text
2. Generated Synthetic Messages:
   - post_number
   - poster_id
   - text
3. Analysis Metrics:
   - Message counts
   - Text lengths
   - Token counts
4. Context Statistics:
   - Messages used
   - Total available messages
   - Tokens used
   - Maximum available tokens
5. Generation Metadata:
   - Model used
   - Generation settings
   - Timestamp

## Recovery

If the generation process is interrupted:
1. Keep the same configuration file
2. Maintain the same checkpoint directory
3. Run the script again with the same parameters
4. The process will automatically resume from the last successful checkpoint

## Monitoring

The script provides progress information:
- Overall completion percentage
- Current conversation count
- Token usage statistics
- Error reporting
- Checkpoint saves
- Upload status (if pushing to HuggingFace)
