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
import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
from tools.generator_utils import (
    generate_dialogue_from_prompt,
    generate_dataset,
    create_analysis_dataset,
    create_hf_dataset
)

"""
Usage:
This script generates synthetic dialogues based on a source dataset and saves them locally and/or to HuggingFace Hub.

Parameters:
--config_file: Path to the JSON configuration file
    The configuration file containing all necessary parameters for dialogue generation

Examples:
# Basic usage with defaults
python generate_dialogues.py prompts/generation_config.json

# Specify all parameters
python generate_dialogues.py prompts/generation_config.json
"""

def load_config(config_file: str) -> dict:
    """Load and validate configuration from JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_keys = ['prompt_config', 'generation_config', 'dataset_config', 'runtime_config']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    return config

def load_prompt_template(template_file: str) -> str:
    """Load prompt template from file."""
    with open(template_file, 'r', encoding='utf-8') as f:
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
    
    checkpoint = {
        'generated_dataset': generated_dataset,
        'processed_ids': list(processed_ids)
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic dialogues using configuration file')
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )
    return parser.parse_args()

def main():
    # Load configuration
    args = parse_args()
    config = load_config(args.config_file)
    
    # Extract configurations
    prompt_config = config['prompt_config']
    generation_config = config['generation_config']
    dataset_config = config['dataset_config']
    runtime_config = config['runtime_config']
    
    # Ensure checkpoint directory exists
    os.makedirs(runtime_config['checkpoint_dir'], exist_ok=True)
    print(f"\nUsing checkpoint directory: {runtime_config['checkpoint_dir']}")
    
    # Load dataset
    dataset = load_dataset(dataset_config['source_dataset'])
    df = dataset['train'].to_pandas()
    conversations = dict(zip(df['conversation_id'], df['messages']))

    # Load checkpoint if exists
    generated_dataset, processed_ids = load_checkpoint(runtime_config['checkpoint_dir'])
    
    # Load prompt template and store its content
    PROMPT_TEMPLATE = load_prompt_template(prompt_config['template_file'])
    prompt_config['template_content'] = PROMPT_TEMPLATE  # Store actual prompt content

    # Get remaining conversations to process
    remaining_ids = set(conversations.keys()) - processed_ids
    remaining_conversations = {k: conversations[k] for k in remaining_ids}
    
    # Calculate how many more conversations we need
    conversations_needed = dataset_config['num_conversations'] - len(generated_dataset)
    
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
                # Format context for single conversation
                context = f"{messages[0]['poster_id']}: {messages[0]['text']}"
                
                # Generate dialogue
                generated_dialogue = generate_dialogue_from_prompt(
                    prompt=PROMPT_TEMPLATE.format(context=context),
                    generation_config=generation_config,
                    vllm_url=runtime_config['vllm_url'],
                    headers=runtime_config['headers']
                )
                
                if generated_dialogue:
                    generated_dataset[conv_id] = {
                        'original_messages': messages,
                        'generated_output': generated_dialogue,
                        'metadata': {'model': generation_config['model']}
                    }
                    processed_ids.add(conv_id)
                    current_count += 1
                    pbar.update(1)
                    
                    # Checkpoint every 100 conversations
                    if current_count % 100 == 0:
                        save_checkpoint(runtime_config['checkpoint_dir'], generated_dataset, processed_ids)
                        print(f"\nCheckpoint saved at {current_count} conversations")
                
            except Exception as e:
                print(f"\nError processing conversation {conv_id}: {str(e)}")
                continue
        
        pbar.close()
        
        # Final checkpoint
        save_checkpoint(runtime_config['checkpoint_dir'], generated_dataset, processed_ids)

    # Create metadata dictionary with all configurations including prompt content
    metadata = {
        "generation_config": generation_config,
        "dataset_config": dataset_config,
        "runtime_config": runtime_config,
        "prompt_config": prompt_config,  # Now includes the template_content
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Create analysis dataset
    model_name = generation_config["model"]
    analysis_dataset = create_analysis_dataset(generated_dataset, model_name)

    # Convert to HuggingFace Dataset with metadata
    hf_dataset = create_hf_dataset(
        analysis_dataset=analysis_dataset,
        split_name="train",
        add_metadata=True,
        dataset_metadata=metadata
    )

    # Save dataset locally
    output_dir = f'data/synthetic_conversations_{len(generated_dataset)}'
    hf_dataset.save_to_disk(output_dir)
    print(f"\nDataset saved locally to: {output_dir}")

    # Push to HuggingFace Hub if requested
    if dataset_config['push_to_hub']:
        hf_dataset.push_to_hub(
            dataset_config['output_dataset_name'],
            private=dataset_config['private'],
            token=True
        )
        print(f"\nDataset pushed to HuggingFace Hub as: {dataset_config['output_dataset_name']}")

    # Print some sample data
    df = hf_dataset["train"].to_pandas()
    print("\nSample conversation:")
    print(df.iloc[0]["synthetic_messages"])

if __name__ == "__main__":
    main() 