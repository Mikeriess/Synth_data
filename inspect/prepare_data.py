from datasets import load_dataset
import json
import os
from pathlib import Path

def prepare_inspection_data(dataset_name: str, output_file: str = "conversations.json"):
    """
    Prepare dataset for the inspection interface.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        output_file: Path to save the JSON file
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    output_path = script_dir / output_file
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Convert to required format
    conversations = []
    current_conv = None
    
    # First, let's print the available columns to debug
    print("Available columns:", dataset['train'].column_names)
    
    for item in dataset['train']:
        if current_conv is None or current_conv['conversation_id'] != item['conversation_id']:
            if current_conv is not None:
                conversations.append(current_conv)
            current_conv = {
                'conversation_id': item['conversation_id'],
                'orig_messages': [],
                'synthetic_messages': []
            }
        
        # The column names match exactly what we need
        current_conv['orig_messages'] = item['orig_messages']
        current_conv['synthetic_messages'] = item['synthetic_messages']
    
    # Add last conversation
    if current_conv is not None:
        conversations.append(current_conv)
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(conversations)} conversations for inspection")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    prepare_inspection_data("mikeriess/LM_dialogues1") 