from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

def prepare_conversation_dataset(
    df: pd.DataFrame,
    text_field: str,
    conversation_id_col: str = 'conversation_id',
    message_features: Optional[List[str]] = None,
    conversation_features: Optional[List[str]] = None,
    include_token_stats: bool = True
) -> List[Dict]:
    """
    Prepare conversations for HuggingFace dataset format with customizable fields.
    Token statistics are only included at conversation level.
    
    Args:
        df: DataFrame with conversation data
        text_field: Column name containing the text to use (e.g. 'anonymized_text')
        conversation_id_col: Column name for conversation IDs
        message_features: List of column names to include for each message
        conversation_features: List of column names to include at conversation level
        include_token_stats: Whether to include token statistics (if available)
    
    Returns:
        List of formatted conversations
    """
    # Default features if none specified
    default_msg_features = ['post_number', 'poster_id', 'post_time']
    default_conv_features = ['forum_name', 'topic_title', 'start_time']
    
    # Use defaults if not specified
    message_features = message_features or default_msg_features
    conversation_features = conversation_features or default_conv_features
    
    formatted_data = []
    
    # Calculate conversation-level token statistics first
    if include_token_stats and f'{text_field}_tokens' in df.columns:
        token_stats = df.groupby(conversation_id_col)[f'{text_field}_tokens'].agg([
            ('total_tokens', 'sum'),
            ('mean_tokens_per_message', 'mean'),
            ('min_tokens_in_conversation', 'min'),
            ('max_tokens_in_conversation', 'max'),
            ('std_tokens', 'std')
        ]).round(2)
    
    # Group by conversation_id
    for conv_id, conv_df in df.sort_values([conversation_id_col, 'post_number']).groupby(conversation_id_col):
        # Initialize conversation data with ID and number of messages
        conv_data = {
            'conversation_id': int(conv_id),
            'num_messages': len(conv_df),
            'messages': []
        }
        
        # Add requested conversation-level features
        for feature in conversation_features:
            if feature in conv_df.columns:
                # Take first value for conversation-level features
                value = conv_df[feature].iloc[0]
                conv_data[feature] = value
        
        # Add comprehensive token statistics if available and requested
        if include_token_stats and f'{text_field}_tokens' in df.columns:
            stats = token_stats.loc[conv_id]
            conv_data.update({
                'total_tokens': int(stats['total_tokens']),
                'mean_tokens_per_message': float(stats['mean_tokens_per_message']),
                'min_tokens_in_conversation': int(stats['min_tokens_in_conversation']),
                'max_tokens_in_conversation': int(stats['max_tokens_in_conversation']),
                'std_tokens': float(stats['std_tokens'])
            })
        
        # Add each message in conversation
        for _, msg in conv_df.iterrows():
            message = {
                # Always include the specified text field
                'text': msg[text_field]
            }
            
            # Add requested message-level features
            for feature in message_features:
                if feature in msg and not feature.endswith('_tokens'):  # Skip token-related features
                    value = msg[feature]
                    # Convert basic types to ensure HF compatibility
                    if isinstance(value, (int, np.integer)):
                        value = int(value)
                    elif isinstance(value, (float, np.floating)):
                        value = float(value)
                    message[feature] = value
            
            conv_data['messages'].append(message)
            
        formatted_data.append(conv_data)
    
    return formatted_data

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
        
        # Token statistics if available
        if 'total_tokens' in formatted_data[0]:
            total_tokens = sum(conv['total_tokens'] for conv in formatted_data)
            avg_tokens = np.mean([conv['total_tokens'] for conv in formatted_data])
            max_tokens = max(conv['total_tokens'] for conv in formatted_data)
            min_tokens = min(conv['total_tokens'] for conv in formatted_data)
            
            print("\nToken Statistics:")
            print(f"Total tokens in dataset: {total_tokens:,}")
            print(f"Average tokens per conversation: {avg_tokens:.1f}")
            print(f"Min tokens in a conversation: {min_tokens}")
            print(f"Max tokens in a conversation: {max_tokens}")