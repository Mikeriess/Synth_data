from typing import Dict
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

def calculate_token_stats(
    df: pd.DataFrame,
    text_field: str,
    conversation_id_col: str = 'conversation_id'
) -> pd.DataFrame:
    """
    Calculate token statistics for conversations.
    
    Args:
        df: DataFrame with conversation data
        text_field: Column name containing the text to use
        conversation_id_col: Column name for conversation IDs
        
    Returns:
        DataFrame with token statistics per conversation
    """
    token_col = f'{text_field}_tokens'
    if token_col not in df.columns:
        return None
        
    return df.groupby(conversation_id_col)[token_col].agg([
        ('total_tokens', 'sum'),
        ('mean_tokens_per_message', 'mean'),
        ('min_tokens_in_conversation', 'min'),
        ('max_tokens_in_conversation', 'max'),
        ('std_tokens', 'std')
    ]).round(2)

def print_token_stats(formatted_data: list) -> None:
    """Print token statistics for the dataset."""
    if not formatted_data or 'total_tokens' not in formatted_data[0]:
        return
        
    total_tokens = sum(conv['total_tokens'] for conv in formatted_data)
    avg_tokens = np.mean([conv['total_tokens'] for conv in formatted_data])
    max_tokens = max(conv['total_tokens'] for conv in formatted_data)
    min_tokens = min(conv['total_tokens'] for conv in formatted_data)
    
    print("\nToken Statistics:")
    print(f"Total tokens in dataset: {total_tokens:,}")
    print(f"Average tokens per conversation: {avg_tokens:.1f}")
    print(f"Min tokens in a conversation: {min_tokens}")
    print(f"Max tokens in a conversation: {max_tokens}") 