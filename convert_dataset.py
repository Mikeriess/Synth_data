from tools.prepare_data import load_dialogue_dataset, prepare_sft_dataset, prepare_dialogue_dataset
from datasets import Dataset
import os

def convert_dataset(dataset_name: str, output_path: str):
    """Convert HuggingFace dataset to ChatML format for Axolotl training
    
    Args:
        dataset_name: Name of the HuggingFace dataset to load
        output_path: Path to save the converted dataset
    """
    print(f"Loading dataset: {dataset_name}")
    
    # Use existing functions from prepare_data.py to load and convert dataset
    chatml_data = prepare_dialogue_dataset(dataset_name=dataset_name)
    
    # Convert to format suitable for training
    converted_dataset = prepare_sft_dataset(chatml_data)
    
    # Save dataset
    print(f"Saving converted dataset to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    converted_dataset.save_to_disk(output_path)
    
    # Print sample to verify format
    print("\nSample from converted dataset:")
    sample = converted_dataset[0]
    print(f"Conversation ID: {sample['conversation_id']}")
    print(f"Language: {sample['language']}")
    print("\nMessages:")
    for msg in sample['messages']:
        print(f"{msg['role']}: {msg['content']}\n")
    
    print("Conversion complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert dialogue dataset to ChatML format')
    parser.add_argument('--dataset', type=str, default="mikeriess/LM_dialogues1",
                      help='HuggingFace dataset name (default: mikeriess/LM_dialogues1)')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save converted dataset')
    
    args = parser.parse_args()
    convert_dataset(args.dataset, args.output) 