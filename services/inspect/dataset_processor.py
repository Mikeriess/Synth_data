import os
import json
import argparse
from datasets import load_dataset
import shutil
import requests
import urllib.parse
import subprocess

# Import the huggingface_hub library for direct downloads
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("huggingface_hub library not available. Installing it would improve file downloads.")

def process_dataset(dataset_id, config=None, split="train", max_samples=None, output_dir="data"):
    """
    Download and process a dataset from Hugging Face.
    
    Args:
        dataset_id (str): The Hugging Face dataset ID
        config (str, optional): Dataset configuration
        split (str): Dataset split (train, validation, test)
        max_samples (int, optional): Maximum number of samples to process
        output_dir (str): Directory to save the processed data
        
    Returns:
        str: Path to the processed data file
    """
    print(f"Loading dataset: {dataset_id} (config: {config}, split: {split})")
    
    # Load the dataset
    if config:
        dataset = load_dataset(dataset_id, config, split=split)
    else:
        dataset = load_dataset(dataset_id, split=split)
    
    # Limit samples if specified
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert dataset to the format expected by the annotation tool
    processed_data = []
    
    # Print dataset features to help with debugging
    print(f"Dataset features: {dataset.features}")
    
    for i, item in enumerate(dataset):
        # Convert item to dict for easier access
        item_dict = dict(item)
        
        # This conversion will depend on the exact format needed by your annotation tool
        # For conversation datasets, try to extract the relevant fields
        if 'orig_messages' in item_dict and 'synthetic_messages' in item_dict:
            # If the dataset already has the right format, use it directly
            processed_item = {
                'id': f"{dataset_id.replace('/', '_')}_{i}",
                'conversation_id': item_dict.get('conversation_id', f"conv_{i}"),
                'orig_messages': item_dict['orig_messages'],
                'synthetic_messages': item_dict['synthetic_messages']
            }
            
            # Extract context statistics if available
            context_fields = [
                'context_msg_used', 'context_msg_available',
                'context_tokens_used', 'context_tokens_available'
            ]
            
            for field in context_fields:
                if field in item_dict:
                    processed_item[field] = item_dict[field]
            
            # If context fields aren't directly available, try to extract from metadata
            if 'metadata' in item_dict and isinstance(item_dict['metadata'], dict):
                metadata = item_dict['metadata']
                for field in context_fields:
                    if field in metadata and field not in processed_item:
                        processed_item[field] = metadata[field]
        else:
            # Otherwise, create a generic item with the text content
            text = item_dict.get('text', '') or item_dict.get('sentence', '') or str(item_dict)
            processed_item = {
                'id': f"{dataset_id.replace('/', '_')}_{i}",
                'text': text
            }
        
        processed_data.append(processed_item)
    
    # Save processed data to a JSON file
    safe_dataset_id = dataset_id.replace('/', '_')
    output_file = os.path.join(output_dir, f"{safe_dataset_id}_{split}.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # Define file paths
    prompt_dest = os.path.join(output_dir, "dialogue_prompt.txt")
    config_dest = os.path.join(output_dir, "generation_config.json")
    
    # First check if local files exist
    local_prompt = os.path.join("prompts", "dialogue_prompt.txt")
    local_config = os.path.join("configs", "generation_config.json")
    
    prompt_found = False
    config_found = False
    
    # Try to copy from local files first
    if os.path.exists(local_prompt):
        shutil.copy2(local_prompt, prompt_dest)
        print(f"Copied prompt file from local directory to {prompt_dest}")
        prompt_found = True
    
    if os.path.exists(local_config):
        shutil.copy2(local_config, config_dest)
        print(f"Copied config file from local directory to {config_dest}")
        config_found = True
    
    # Try to download using huggingface_hub if available
    if HF_HUB_AVAILABLE and (not prompt_found or not config_found):
        try:
            # Try to download the prompt file
            if not prompt_found:
                try:
                    # Try different possible paths for the prompt file
                    prompt_paths = [
                        "prompts/dialogue_prompt.txt",
                        "dialogue_prompt.txt"
                    ]
                    
                    for path in prompt_paths:
                        try:
                            print(f"Trying to download prompt from path: {path}")
                            downloaded_file = hf_hub_download(
                                repo_id=dataset_id,
                                filename=path,
                                repo_type="dataset"
                            )
                            # Copy the downloaded file to the destination
                            shutil.copy2(downloaded_file, prompt_dest)
                            print(f"Successfully downloaded prompt file to {prompt_dest}")
                            prompt_found = True
                            break
                        except Exception as e:
                            print(f"Failed to download prompt from path {path}: {e}")
                except Exception as e:
                    print(f"Error downloading prompt file with huggingface_hub: {e}")
            
            # Try to download the config file
            if not config_found:
                try:
                    # Try different possible paths for the config file
                    config_paths = [
                        "configs/generation_config.json",
                        "generation_config.json"
                    ]
                    
                    for path in config_paths:
                        try:
                            print(f"Trying to download config from path: {path}")
                            downloaded_file = hf_hub_download(
                                repo_id=dataset_id,
                                filename=path,
                                repo_type="dataset"
                            )
                            # Copy the downloaded file to the destination
                            shutil.copy2(downloaded_file, config_dest)
                            print(f"Successfully downloaded config file to {config_dest}")
                            config_found = True
                            break
                        except Exception as e:
                            print(f"Failed to download config from path {path}: {e}")
                except Exception as e:
                    print(f"Error downloading config file with huggingface_hub: {e}")
        except Exception as e:
            print(f"Error using huggingface_hub: {e}")
    
    # If still not found, try with wget and requests as fallbacks
    if not prompt_found:
        # Try wget first
        try:
            prompt_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/prompts/dialogue_prompt.txt"
            print(f"Attempting to download prompt file from: {prompt_url}")
            result = subprocess.run(["wget", "-q", "-O", prompt_dest, prompt_url], check=False)
            if result.returncode == 0 and os.path.exists(prompt_dest) and os.path.getsize(prompt_dest) > 0:
                print(f"Successfully downloaded prompt file to {prompt_dest}")
                prompt_found = True
        except Exception as e:
            print(f"Error downloading prompt file with wget: {e}")
    
    if not config_found:
        try:
            config_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/configs/generation_config.json"
            print(f"Attempting to download config file from: {config_url}")
            result = subprocess.run(["wget", "-q", "-O", config_dest, config_url], check=False)
            if result.returncode == 0 and os.path.exists(config_dest) and os.path.getsize(config_dest) > 0:
                print(f"Successfully downloaded config file to {config_dest}")
                config_found = True
        except Exception as e:
            print(f"Error downloading config file with wget: {e}")
    
    # If still not found, create placeholder files
    if not prompt_found:
        with open(prompt_dest, "w", encoding='utf-8') as f:
            f.write(f"No prompt file found for dataset: {dataset_id}\n\nPlease check the repository at https://huggingface.co/datasets/{dataset_id} for the prompt file.")
        print(f"Created placeholder prompt file at {prompt_dest}")
    
    if not config_found:
        with open(config_dest, "w", encoding='utf-8') as f:
            f.write(json.dumps({"message": f"No config file found for dataset: {dataset_id}"}, indent=2))
        print(f"Created placeholder config file at {config_dest}")
    
    print(f"Processed {len(processed_data)} samples. Saved to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Hugging Face dataset for annotation")
    parser.add_argument("--dataset_id", required=True, help="Hugging Face dataset ID")
    parser.add_argument("--config", help="Dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split (train, validation, test)")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--output_dir", default="data", help="Directory to save the processed data")
    
    args = parser.parse_args()
    process_dataset(args.dataset_id, args.config, args.split, args.max_samples, args.output_dir) 