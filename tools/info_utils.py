from typing import List, Dict

def print_dataset_info(formatted_data: List[Dict]) -> None:
    """
    Print information about the formatted dataset.
    
    Args:
        formatted_data: List of formatted conversations
    """
    print("Dataset Summary:")
    print(f"Number of conversations: {len(formatted_data)}")
    
    if formatted_data:
        # Get features from first conversation
        conv_features = set(formatted_data[0].keys()) - {'messages'}
        print("\nConversation-level features:")
        print(", ".join(sorted(conv_features)))
        
        # Get message features from first message
        if formatted_data[0]['messages']:
            msg_features = set(formatted_data[0]['messages'][0].keys())
            print("\nMessage-level features:")
            print(", ".join(sorted(msg_features)))
        
        # Basic statistics
        num_messages = sum(len(conv['messages']) for conv in formatted_data)
        print(f"\nTotal messages: {num_messages}")
        print(f"Average messages per conversation: {num_messages/len(formatted_data):.1f}") 