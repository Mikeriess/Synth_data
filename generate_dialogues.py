import pickle
import json
import requests
from typing import List, Dict
import random
from tqdm import tqdm
import re
import argparse
import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tools.generator_utils import (
    generate_dialogue_from_prompt,
    generate_dataset,
    create_analysis_dataset,
    create_hf_dataset,
    parse_generated_dialogue_to_messages
)
from huggingface_hub import HfApi
import pandas as pd

"""
Usage:
This script generates synthetic dialogues based on a source dataset and saves them locally and/or to HuggingFace Hub.

Parameters:
--prompt_file: Path to the prompt template file (default: 'prompts/dialogue_prompt.txt')
    The template file containing the prompt structure for dialogue generation

--num_conversations: Number of conversations to generate (default: 1000)
    Total number of synthetic dialogues to create

--dataset_name: Name for the dataset when pushing to HuggingFace Hub (default: 'mikeriess/LM_dialogues1')
    Format: 'username/dataset-name'

--push_to_hub: Flag to enable pushing to HuggingFace Hub (default: False)
    If set, uploads the dataset to HF Hub (requires authentication)

--checkpoint_dir: Directory for storing checkpoints (default: 'checkpoints')
    Saves progress every 100 conversations and enables resuming interrupted runs

Examples:
# Basic usage with defaults
python generate_dialogues.py

# Specify all parameters
python generate_dialogues.py \
    --prompt_file prompts/dialogue_prompt.txt \
    --num_conversations 500 \
    --dataset_name "mikeriess/my-dataset" \
    --push_to_hub \
    --checkpoint_dir "checkpoints/run1"

# Generate 100 conversations without pushing to hub
python generate_dialogues.py --num_conversations 100
"""

def load_prompt_template(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def load_checkpoint(checkpoint_dir: str) -> tuple[dict, set]:
    """Load existing generated dialogues and processed conversation IDs."""
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint['generated_dataset'], set(checkpoint['processed_ids'])
    return {}, set()

def save_checkpoint(checkpoint_dir: str, generated_dataset: dict, processed_ids: set):
    """Save current progress to checkpoint file."""
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.json"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processed_ids_list = [int(id_) for id_ in processed_ids]
    
    json_safe_dataset = {}
    for conv_id, data in generated_dataset.items():
        conv_id_str = str(conv_id)
        
        try:
            # Function to safely convert to int, handling NaN
            def safe_int(value):
                if pd.isna(value):  # Check for NaN
                    print(f"Warning: Found NaN value in conversation {conv_id}")
                    return 0  # or some default value
                return int(value)
            
            json_safe_dataset[conv_id_str] = {
                'original_messages': [
                    {
                        'post_number': safe_int(msg.get('post_number', 0)),
                        'poster_id': safe_int(msg.get('poster_id', 0)),
                        'text': str(msg.get('text', ''))
                    }
                    for msg in data['original_messages']
                ],
                'generated_output': str(data['generated_output']),
                'parsed_messages': [
                    {
                        'post_number': safe_int(msg.get('post_number', 0)),
                        'poster_id': safe_int(msg.get('poster_id', 0)),
                        'text': str(msg.get('text', ''))
                    }
                    for msg in data['parsed_messages']
                ],
                'metadata': {
                    'model': str(data['metadata']['model'])
                }
            }
        except Exception as e:
            print(f"Warning: Error processing conversation {conv_id}: {str(e)}")
            continue
    
    checkpoint = {
        'generated_dataset': json_safe_dataset,
        'processed_ids': processed_ids_list
    }
    
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def upload_config_files(dataset_name: str, config_path: str, prompt_path: str):
    """Upload configuration and prompt files to dataset repository."""
    api = HfApi()
    
    # Upload config file
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="configs/generation_config.json",
        repo_id=dataset_name,
        repo_type="dataset"
    )
    
    # Upload prompt file
    api.upload_file(
        path_or_fileobj=prompt_path,
        path_in_repo="prompts/dialogue_prompt.txt",
        repo_id=dataset_name,
        repo_type="dataset"
    )

def create_and_upload_readme(dataset_name: str, config: dict):
    """Create and upload README.md containing the configuration."""
    try:
        # Create README content (raw JSON, no markdown)
        readme_content = json.dumps(config, indent=2)
        
        # Create temporary README file
        readme_path = Path("README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset"
        )
        
        # Clean up temporary file
        readme_path.unlink()
        
    except Exception as e:
        print(f"\nWarning: Failed to upload README: {str(e)}")

def upload_intermediate_dataset(generated_dataset: dict, config: dict, current_count: int):
    """Upload intermediate dataset to HuggingFace Hub."""
    try:
        # Create analysis dataset
        model_name = config['generation_config']['model']
        analysis_dataset = create_analysis_dataset(generated_dataset, model_name)
        
        # Convert to HF dataset
        hf_dataset = create_hf_dataset(
            analysis_dataset=analysis_dataset,
            split_name="train",
            add_metadata=True
        )
        
        # Push to HF Hub with version tag
        hf_dataset.push_to_hub(
            config['dataset_config']['output_dataset_name'],
            private=config['dataset_config']['private'],
            token=True,
            commit_message=f"Intermediate upload - {current_count} conversations"
        )
        print(f"\nIntermediate dataset ({current_count} conversations) pushed to HuggingFace Hub")
        
        # Upload README with configuration
        create_and_upload_readme(
            config['dataset_config']['output_dataset_name'],
            config
        )
        
    except Exception as e:
        print(f"\nWarning: Failed to upload intermediate dataset: {str(e)}")

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    
    # Ensure checkpoint directory exists
    os.makedirs(config['files']['checkpoint_dir'], exist_ok=True)
    print(f"\nUsing checkpoint directory: {config['files']['checkpoint_dir']}")
    
    # Load dataset
    dataset = load_dataset(config['dataset_config']['source_dataset'])
    df = dataset['train'].to_pandas()
    conversations = dict(zip(df['conversation_id'], df['messages']))

    # Load checkpoint if exists
    generated_dataset, processed_ids = load_checkpoint(config['files']['checkpoint_dir'])
    
    # Load prompt template
    PROMPT_TEMPLATE = load_prompt_template(config['files']['prompt_file'])

    # Get remaining conversations to process
    remaining_ids = set(conversations.keys()) - processed_ids
    remaining_conversations = {k: conversations[k] for k in remaining_ids}
    
    # Calculate how many more conversations we need
    conversations_needed = config['dataset_config']['num_conversations'] - len(generated_dataset)
    
    if conversations_needed > 0:
        print(f"\nGenerating {conversations_needed} more conversations...")
        print(f"Already generated: {len(generated_dataset)} conversations")
        
        # Generate remaining conversations with checkpointing
        pbar = tqdm(total=conversations_needed)
        current_count = 0
        
        for conv_id, messages in remaining_conversations.items():
            if current_count >= conversations_needed:
                break
                
            try:
                context = f"{messages[0]['poster_id']}: {messages[0]['text']}"
                
                generated_dialogue = generate_dialogue_from_prompt(
                    prompt=PROMPT_TEMPLATE.format(context=context),
                    generation_config={
                        **config['generation_config'],
                        'api_config': config['api_config']  # Include API configuration
                    }
                )
                
                if generated_dialogue:
                    # Parse the generated dialogue into messages
                    parsed_messages = parse_generated_dialogue_to_messages(generated_dialogue)
                    
                    generated_dataset[conv_id] = {
                        'original_messages': messages,
                        'generated_output': generated_dialogue,
                        'parsed_messages': parsed_messages,
                        'metadata': {'model': config['generation_config']['model']}
                    }
                    processed_ids.add(conv_id)
                    current_count += 1
                    pbar.update(1)
                    
                    if current_count % 100 == 0:
                        # Save local checkpoint
                        save_checkpoint(config['files']['checkpoint_dir'], generated_dataset, processed_ids)
                        print(f"\nCheckpoint saved at {current_count} conversations")
                        
                        # Upload intermediate dataset if push_to_hub is enabled
                        if config['dataset_config']['push_to_hub']:
                            upload_intermediate_dataset(generated_dataset, config, current_count)
                
            except Exception as e:
                print(f"\nError processing conversation {conv_id}: {str(e)}")
                continue
        
        pbar.close()
        save_checkpoint(config['files']['checkpoint_dir'], generated_dataset, processed_ids)

    # Create and save dataset
    model_name = config['generation_config']['model']
    analysis_dataset = create_analysis_dataset(generated_dataset, model_name)
    hf_dataset = create_hf_dataset(
        analysis_dataset=analysis_dataset,
        split_name="train",
        add_metadata=True
    )

    # Save dataset locally
    output_dir = f'data/synthetic_conversations_{len(generated_dataset)}'
    hf_dataset.save_to_disk(output_dir)
    print(f"\nDataset saved locally to: {output_dir}")

    # Push to HuggingFace Hub if requested
    if config['dataset_config']['push_to_hub']:
        hf_dataset.push_to_hub(
            config['dataset_config']['output_dataset_name'],
            private=config['dataset_config']['private'],
            token=True
        )
        print(f"\nDataset pushed to HuggingFace Hub as: {config['dataset_config']['output_dataset_name']}")
        
        # Upload config and prompt files
        upload_config_files(
            config['dataset_config']['output_dataset_name'],
            config_path,
            config['files']['prompt_file']
        )
        
        # Upload README with configuration
        create_and_upload_readme(
            config['dataset_config']['output_dataset_name'],
            config
        )
        
        print("\nConfiguration, prompt files, and README uploaded to dataset repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/generation_config.json',
                      help='Path to generation configuration file')
    args = parser.parse_args()
    main(args.config) 