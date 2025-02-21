from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from .token_utils import calculate_token_stats

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
    if include_token_stats:
        token_stats = calculate_token_stats(df, text_field, conversation_id_col)
    
    # Group by conversation_id
    for conv_id, conv_df in df.sort_values([conversation_id_col, 'post_number']).groupby(conversation_id_col):
        # Initialize conversation data
        conv_data = {
            'conversation_id': int(conv_id),
            'num_messages': len(conv_df),
            'messages': []
        }
        
        # Add requested conversation-level features
        for feature in conversation_features:
            if feature in conv_df.columns:
                value = conv_df[feature].iloc[0]
                conv_data[feature] = value
        
        # Add token statistics if available
        if include_token_stats and token_stats is not None:
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
            message = {'text': msg[text_field]}
            
            # Add requested message-level features
            for feature in message_features:
                if feature in msg and not feature.endswith('_tokens'):
                    value = msg[feature]
                    if isinstance(value, (int, np.integer)):
                        value = int(value)
                    elif isinstance(value, (float, np.floating)):
                        value = float(value)
                    message[feature] = value
            
            conv_data['messages'].append(message)
            
        formatted_data.append(conv_data)
    
    return formatted_data 