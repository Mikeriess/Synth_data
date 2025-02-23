from datasets import load_dataset, Dataset, DatasetDict
from typing import Dict, List, Optional, Union
import pandas as pd

def load_dialogue_dataset(dataset_name: str = "mikeriess/LM_dialogues1") -> pd.DataFrame:
    """
    Load the dialogue dataset and convert to pandas DataFrame.
    
    Args:
        dataset_name: Name of the dataset to load from HuggingFace
        
    Returns:
        DataFrame containing the dialogue data
    """
    dataset = load_dataset(dataset_name)
    return dataset['train'].to_pandas()

def convert_to_chatml(
    df: pd.DataFrame,
    language: str = "Danish"
) -> Dict[str, Dict]:
    """
    Convert dialogue dataset to ChatML format.
    
    Args:
        df: DataFrame containing dialogue data
        language: Language of the conversations
        
    Returns:
        Dictionary in ChatML format with structure:
        {
            "synthetic_id": {
                "messages": [
                    {"role": "user/assistant", "content": "message text"},
                    ...
                ],
                "language": language
            }
        }
    """
    chatml_data = {}
    
    for idx, row in df.iterrows():
        # Create messages list
        messages = []
        for message in row['synthetic_messages']:
            # Determine role based on user field
            # Assuming user 1 is the questioner (user) and user 2 is the answerer (assistant)
            role = "user" if message['user'] == 1 else "assistant"
            
            messages.append({
                "role": role,
                "content": message['text']
            })
        
        # Generate a unique ID
        question_id = f"synthetic_{idx}"
        
        # Store in chatml format
        chatml_data[question_id] = {
            "messages": messages,
            "language": language
        }
    
    return chatml_data

def prepare_dialogue_dataset(
    dataset_name: str = "mikeriess/LM_dialogues1",
    language: str = "Danish"
) -> Dict[str, Dict]:
    """
    Load and prepare dialogue dataset in ChatML format.
    
    Args:
        dataset_name: Name of the dataset to load from HuggingFace
        language: Language of the conversations
        
    Returns:
        Dictionary in ChatML format
    """
    # Load dataset
    df = load_dialogue_dataset(dataset_name)
    
    # Convert to ChatML format
    return convert_to_chatml(df, language)

def print_sample_conversation(chatml_data: Dict[str, Dict], sample_idx: int = 1) -> None:
    """
    Print a sample conversation from the ChatML dataset.
    
    Args:
        chatml_data: Dictionary in ChatML format
        sample_idx: Index of the sample to print (default: 1)
    """
    sample_id = list(chatml_data.keys())[sample_idx]
    print("Sample conversation:")
    print(f"\nQuestion ID: {sample_id}")
    print(f"Language: {chatml_data[sample_id]['language']}")
    print("\nMessages:")
    for msg in chatml_data[sample_id]['messages']:
        print(f"{msg['role']}: {msg['content']}\n")

def prepare_sft_dataset(chatml_data):
    """
    Convert ChatML format data to a format suitable for SFTTrainer.
    
    Args:
        chatml_data: Dictionary of conversations in ChatML format
        
    Returns:
        Dataset: HuggingFace Dataset ready for SFTTrainer
    """
    # Convert conversations to list format
    formatted_data = []
    
    for conv_id, conv_data in chatml_data.items():
        # Combine messages into a single conversation string
        messages = conv_data['messages']
        
        # Format each conversation as a dictionary with the required fields
        formatted_data.append({
            "messages": messages,  # SFTTrainer can handle the ChatML format directly
            "conversation_id": conv_id,
            "language": conv_data['language']
        })
    
    # Convert to pandas DataFrame first
    df = pd.DataFrame(formatted_data)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

if __name__ == "__main__":
    # Example usage
    chatml_data = prepare_dialogue_dataset()
    print_sample_conversation(chatml_data) 