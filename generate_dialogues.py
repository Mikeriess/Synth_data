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
    --num_conversations 100 \
    --dataset_name "mikeriess/lm_dialogues3_gemma9b" \
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
    checkpoint_path = Path(checkpoint_dir) / "state/checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint['generated_dataset'], set(checkpoint['processed_ids'])
    return {}, set()

def validate_conversation(messages: List[Dict]) -> bool:
    """
    Validate conversation messages for NaN values and required fields.
    Returns True if valid, False otherwise.
    """
    try:
        for msg in messages:
            # Check for required fields
            if not all(key in msg for key in ['post_number', 'poster_id', 'text']):
                print(f"Warning: Missing required fields in message: {msg}")
                return False
            
            # Check for NaN values
            if any(pd.isna(msg.get(key)) for key in ['post_number', 'poster_id']):
                print(f"Warning: Found NaN value in message: {msg}")
                return False
            
            # Check for empty or NaN text
            if pd.isna(msg.get('text')) or not str(msg.get('text')).strip():
                print(f"Warning: Empty or NaN text in message: {msg}")
                return False
            
            # Ensure numeric fields are valid
            try:
                int(msg.get('post_number'))
                int(msg.get('poster_id'))
            except (ValueError, TypeError):
                print(f"Warning: Invalid numeric values in message: {msg}")
                return False
                
        return True
    except Exception as e:
        print(f"Warning: Error validating conversation: {str(e)}")
        return False

def safe_int(value, default: int = 0) -> int:
    """Safely convert value to integer, handling various edge cases."""
    try:
        if pd.isna(value):
            return default
        # Try converting to float first to handle string numbers
        return int(float(value))
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to int, using default {default}")
        return default

def save_checkpoint(checkpoint_dir: str, generated_dataset: dict, processed_ids: set):
    """Save current progress to checkpoint file."""
    # Create directory structure
    checkpoint_dir = Path(checkpoint_dir)
    state_dir = checkpoint_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = state_dir / "checkpoint.json"
    
    processed_ids_list = [int(id_) for id_ in processed_ids]
    
    json_safe_dataset = {}
    for conv_id, data in generated_dataset.items():
        conv_id_str = str(conv_id)
        
        try:
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
        # Create a temporary analysis dataset
        model_name = config['generation_config']['model']
        analysis_dataset = []
        for conv_id, data in generated_dataset.items():
            # Extract original and synthetic messages
            orig_messages = data['original_messages']
            synthetic_messages = data['parsed_messages']
            
            # Create analysis entry
            analysis_entry = {
                'model': model_name,
                'conversation_id': conv_id,
                'orig_messages': orig_messages,
                'synthetic_messages': synthetic_messages,
                'orig_message_count': len(orig_messages),
                'synthetic_message_count': len(synthetic_messages),
                'message_count_diff': len(synthetic_messages) - len(orig_messages),
                'orig_total_length': sum(len(msg.get('text', '')) for msg in orig_messages),
                'synthetic_total_length': sum(len(msg.get('text', '')) for msg in synthetic_messages),
                'orig_total_tokens': sum(len(msg.get('text', '').split()) for msg in orig_messages),
                'synthetic_total_tokens': sum(len(msg.get('text', '').split()) for msg in synthetic_messages)
            }
            
            # Add context statistics if available
            if 'metadata' in data and 'context_stats' in data['metadata']:
                context_stats = data['metadata']['context_stats']
                analysis_entry['context_msg_used'] = context_stats.get('messages_used', 0)
                analysis_entry['context_msg_available'] = context_stats.get('total_messages', 0)
                analysis_entry['context_tokens_used'] = context_stats.get('tokens_used', 0)
                analysis_entry['context_tokens_available'] = context_stats.get('max_tokens', 0)
            
            # Add metadata if available
            if 'metadata' in data:
                analysis_entry['metadata'] = data['metadata']
            
            analysis_dataset.append(analysis_entry)
        
        # Create HF dataset
        hf_dataset = create_hf_dataset(
            analysis_dataset=analysis_dataset,
            split_name="train",
            add_metadata=True
        )
        
        # Push to HuggingFace Hub using the final dataset name
        # This will create the dataset if it doesn't exist, or update it if it does
        hf_dataset.push_to_hub(
            config['dataset_config']['output_dataset_name'],
            private=config['dataset_config']['private'],
            token=True,
            commit_message=f"Intermediate update - {current_count} conversations"
        )
        print(f"\nIntermediate dataset ({current_count} conversations) pushed to HuggingFace Hub as: {config['dataset_config']['output_dataset_name']}")
        
    except Exception as e:
        print(f"\nWarning: Failed to upload intermediate dataset: {str(e)}")

def save_local_dataset(checkpoint_dir: str, hf_dataset, num_conversations: int):
    """Save dataset locally in the checkpoint directory."""
    output_dir = Path(checkpoint_dir) / f'synthetic_conversations_{num_conversations}'
    # Convert Path to string before saving
    hf_dataset.save_to_disk(str(output_dir))
    return output_dir

def backup_config_files(checkpoint_dir: str, config_path: str, prompt_path: str):
    """Backup configuration and prompt files to checkpoint directory."""
    config_dir = Path(checkpoint_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file
    config_backup = config_dir / "generation_config.json"
    with open(config_path, 'r') as src, open(config_backup, 'w') as dst:
        json.dump(json.load(src), dst, indent=2)
    
    # Copy prompt file
    prompt_backup = config_dir / "dialogue_prompt.txt"
    with open(prompt_path, 'r') as src, open(prompt_backup, 'w') as dst:
        dst.write(src.read())

def build_context_with_token_limit(messages: List[Dict], prompt_template: str, max_input_tokens: int) -> tuple:
    """
    Build context from messages while respecting token limit.
    
    Args:
        messages: List of message dictionaries
        prompt_template: The prompt template string
        max_input_tokens: Maximum allowed tokens for the entire input
        
    Returns:
        Tuple of (context string, context statistics dictionary)
    """
    # Initialize tokenizer (using a fast tokenizer for efficiency)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    
    # Calculate tokens in the prompt template (with an empty context)
    prompt_tokens = len(tokenizer.encode(prompt_template.format(context="")))
    
    # Calculate available tokens for context
    available_tokens = max_input_tokens - prompt_tokens
    
    if available_tokens <= 0:
        print(f"Warning: Prompt template uses {prompt_tokens} tokens, exceeding the limit of {max_input_tokens}")
        return "", {"messages_used": 0, "total_messages": len(messages), "tokens_used": 0, "max_tokens": 0}
    
    # Build context incrementally
    context_parts = []
    total_context_tokens = 0
    
    for msg in messages:
        # Format this message
        msg_text = f"{msg['poster_id']}: {msg['text']}"
        msg_tokens = len(tokenizer.encode(msg_text))
        
        # Check if adding this message would exceed the limit
        if total_context_tokens + msg_tokens > available_tokens:
            break
        
        # Add message to context
        context_parts.append(msg_text)
        total_context_tokens += msg_tokens
    
    # Join all context parts
    context = "\n".join(context_parts)
    
    # Create context statistics dictionary
    context_stats = {
        "messages_used": len(context_parts),
        "total_messages": len(messages),
        "tokens_used": total_context_tokens,
        "max_tokens": available_tokens
    }
    
    print(f"Using {len(context_parts)}/{len(messages)} messages ({total_context_tokens} tokens) for context")
    return context, context_stats

def create_analysis_dataset(generated_dataset: dict, model_name: str) -> List[Dict]:
    """
    Create a dataset for analysis from the generated dialogues.
    
    Args:
        generated_dataset: Dictionary of generated dialogues
        model_name: Name of the model used for generation
        
    Returns:
        List of dictionaries with analysis data
    """
    analysis_data = []
    
    for conv_id, data in generated_dataset.items():
        # Extract original and synthetic messages
        orig_messages = data['original_messages']
        synthetic_messages = data['parsed_messages']
        
        # Calculate message counts
        orig_message_count = len(orig_messages)
        synthetic_message_count = len(synthetic_messages)
        
        # Calculate total text length
        orig_total_length = sum(len(msg.get('text', '')) for msg in orig_messages)
        synthetic_total_length = sum(len(msg.get('text', '')) for msg in synthetic_messages)
        
        # Calculate total tokens (approximate)
        orig_total_tokens = sum(len(msg.get('text', '').split()) for msg in orig_messages)
        synthetic_total_tokens = sum(len(msg.get('text', '').split()) for msg in synthetic_messages)
        
        # Extract context statistics if available
        context_msg_used = 0
        context_msg_available = 0
        context_tokens_used = 0
        context_tokens_available = 0
        
        if 'metadata' in data and 'context_stats' in data['metadata']:
            context_stats = data['metadata']['context_stats']
            context_msg_used = context_stats.get('messages_used', 0)
            context_msg_available = context_stats.get('total_messages', 0)
            context_tokens_used = context_stats.get('tokens_used', 0)
            context_tokens_available = context_stats.get('max_tokens', 0)
        
        # Create analysis entry
        analysis_entry = {
            'model': model_name,
            'conversation_id': conv_id,
            'orig_messages': orig_messages,
            'synthetic_messages': synthetic_messages,
            'orig_message_count': orig_message_count,
            'synthetic_message_count': synthetic_message_count,
            'message_count_diff': synthetic_message_count - orig_message_count,
            'orig_total_length': orig_total_length,
            'synthetic_total_length': synthetic_total_length,
            'orig_total_tokens': orig_total_tokens,
            'synthetic_total_tokens': synthetic_total_tokens,
            # Add context statistics as top-level fields
            'context_msg_used': context_msg_used,
            'context_msg_available': context_msg_available,
            'context_tokens_used': context_tokens_used,
            'context_tokens_available': context_tokens_available
        }
        
        # Add metadata if available
        if 'metadata' in data:
            analysis_entry['metadata'] = data['metadata']
        
        analysis_data.append(analysis_entry)
    
    return analysis_data

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    
    # Get max input tokens from config (with default fallback)
    max_input_tokens = config.get('generation_config', {}).get('total_input_tokens', 512)
    
    # Setup checkpoint directory structure
    checkpoint_dir = Path(config['files']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup config and prompt files
    backup_config_files(
        checkpoint_dir,
        config_path,
        config['files']['prompt_file']
    )
    print(f"\nConfiguration files backed up to: {checkpoint_dir}/configs")
    
    # Load dataset
    dataset = load_dataset(config['dataset_config']['source_dataset'])
    df = dataset['train'].to_pandas()
    conversations = dict(zip(df['conversation_id'], df['messages']))

    # Load checkpoint if exists
    generated_dataset, processed_ids = load_checkpoint(str(checkpoint_dir))
    
    # Load prompt template
    PROMPT_TEMPLATE = load_prompt_template(config['files']['prompt_file'])

    # Get remaining conversations to process
    remaining_ids = set(conversations.keys()) - processed_ids
    remaining_conversations = {
        k: v for k, v in conversations.items() 
        if k in remaining_ids and validate_conversation(v)
    }
    
    print(f"\nValid conversations to process: {len(remaining_conversations)} out of {len(remaining_ids)} remaining")
    
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
                # Skip if conversation is invalid
                if not validate_conversation(messages):
                    print(f"\nSkipping invalid conversation {conv_id}")
                    continue
                    
                # Build context with token limit
                context, context_stats = build_context_with_token_limit(
                    messages=messages,
                    prompt_template=PROMPT_TEMPLATE,
                    max_input_tokens=max_input_tokens
                )
                
                generated_dialogue = generate_dialogue_from_prompt(
                    prompt=PROMPT_TEMPLATE.format(context=context),
                    generation_config={
                        **config['generation_config'],
                        'api_config': config['api_config']
                    }
                )
                
                if generated_dialogue:
                    # Parse the generated dialogue into messages
                    parsed_messages = parse_generated_dialogue_to_messages(generated_dialogue)
                    
                    generated_dataset[conv_id] = {
                        'original_messages': messages,
                        'generated_output': generated_dialogue,
                        'parsed_messages': parsed_messages,
                        'metadata': {
                            'model': config['generation_config']['model'],
                            'context_stats': context_stats
                        }
                    }
                    processed_ids.add(conv_id)
                    current_count += 1
                    pbar.update(1)
                    
                    if current_count % 100 == 0:
                        # Save local checkpoint
                        save_checkpoint(str(checkpoint_dir), generated_dataset, processed_ids)
                        print(f"\nCheckpoint saved at {current_count} conversations")
                        
                        # Upload intermediate dataset if push_to_hub is enabled
                        if config['dataset_config']['push_to_hub']:
                            upload_intermediate_dataset(generated_dataset, config, current_count)
                
            except Exception as e:
                print(f"\nError processing conversation {conv_id}: {str(e)}")
                continue
        
        pbar.close()
        save_checkpoint(str(checkpoint_dir), generated_dataset, processed_ids)

    # Create and save dataset
    model_name = config['generation_config']['model']
    analysis_dataset = create_analysis_dataset(generated_dataset, model_name)
    
    hf_dataset = create_hf_dataset(
        analysis_dataset=analysis_dataset,
        split_name="train",
        add_metadata=True
    )

    # Save final dataset locally in checkpoint directory
    output_dir = save_local_dataset(
        str(checkpoint_dir),
        hf_dataset,
        len(generated_dataset)
    )
    print(f"\nDataset saved locally to: {output_dir}")

    # Push to HuggingFace Hub if requested
    if config['dataset_config']['push_to_hub']:
        hf_dataset.push_to_hub(
            config['dataset_config']['output_dataset_name'],
            private=config['dataset_config']['private'],
            token=True
        )
        print(f"\nDataset pushed to HuggingFace Hub as: {config['dataset_config']['output_dataset_name']}")
        
        # Upload config and prompt files from checkpoint directory
        upload_config_files(
            config['dataset_config']['output_dataset_name'],
            str(checkpoint_dir / "configs/generation_config.json"),
            str(checkpoint_dir / "configs/dialogue_prompt.txt")
        )
        print("\nConfiguration and prompt files uploaded to dataset repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/generation_config.json',
                      help='Path to generation configuration file')
    args = parser.parse_args()
    main(args.config) 