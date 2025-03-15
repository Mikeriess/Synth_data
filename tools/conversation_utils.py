import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import json
from datasets import Dataset, DatasetDict

def organize_conversations(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Organize forum posts into conversations.
    
    Args:
        df: DataFrame containing forum posts
        
    Returns:
        Dictionary mapping conversation IDs to conversation data
    """
    conversations = {}
    
    # Group posts by topic_id
    for topic_id, topic_posts in df.groupby('topic_id'):
        # Sort posts by timestamp
        topic_posts = topic_posts.sort_values('post_time')
        
        # Create conversation object
        conversation = {
            'metadata': {
                'topic_id': int(topic_id),
                'topic_title': topic_posts.iloc[0]['topic_title'],
                'forum_id': int(topic_posts.iloc[0]['forum_id']),
                'forum_name': topic_posts.iloc[0]['forum_name'],
                'topic_poster': int(topic_posts.iloc[0]['topic_poster']),
                'total_posts': len(topic_posts)
            },
            'posts': []
        }
        
        # Add each post to conversation
        for idx, (_, post) in enumerate(topic_posts.iterrows(), 1):
            # Handle potentially missing text
            post_text = post['post_text'] if pd.notna(post['post_text']) else ''
            
            conversation['posts'].append({
                'text': post_text,
                'post_id': int(post['post_id']) if pd.notna(post['post_id']) else None,
                'post_time': int(post['post_time']) if pd.notna(post['post_time']) else None,
                'poster_id': int(post['poster_id']) if pd.notna(post['poster_id']) else None,
                'post_subject': post['post_subject'] if pd.notna(post['post_subject']) else '',
                'post_number': idx  # Add post number
            })
            
        conversations[int(topic_id)] = conversation
        
    return conversations

def conversations_to_dataframe(conversations: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert conversations dictionary to a pandas DataFrame.
    
    Args:
        conversations: Dictionary of conversations
        
    Returns:
        DataFrame with one row per post, including conversation metadata
    """
    rows = []
    
    for conv_id, conversation in conversations.items():
        metadata = conversation['metadata']
        
        for post in conversation['posts']:
            # Create a unique row for each post with all metadata
            row = {
                'conversation_id': int(conv_id),
                'post_number': int(post['post_number']),
                'post_id': int(post['post_id']) if post['post_id'] is not None else None,
                'poster_id': int(post['poster_id']) if post['poster_id'] is not None else None,
                'post_time': post['post_time'],
                'forum_name': metadata['forum_name'],
                'topic_title': metadata['topic_title'],
                'total_posts_in_conversation': int(metadata['total_posts']),
                'original_text': post['original_text'],
                'normalized_text': post['normalized_text'],
                'anonymized_text': post['anonymized_text']
            }
            rows.append(row)
            
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by conversation_id and post_number
    df = df.sort_values(['conversation_id', 'post_number'])
    
    return df

def conversations_to_hf_format(conversations: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert conversations to format suitable for HuggingFace dataset.
    """
    formatted_data = []
    
    for conv_id, conversation in conversations.items():
        metadata = conversation['metadata']
        
        # Create conversation messages
        messages = []
        for post in conversation['posts']:
            # Add system message for first post
            if post['post_number'] == 1:
                messages.append({
                    "role": "system",
                    "content": f"Forum: {metadata['forum_name']}\nTopic: {metadata['topic_title']}"
                })
            
            # Add post as user/assistant message
            role = "user" if post['post_number'] % 2 == 1 else "assistant"
            messages.append({
                "role": role,
                "content": post['anonymized_text']
            })
        
        # Add formatted conversation
        formatted_data.append({
            "id": str(conv_id),
            "messages": messages,
            "metadata": {
                "forum_name": metadata['forum_name'],
                "topic_title": metadata['topic_title'],
                "total_posts": metadata['total_posts']
            }
        })
    
    return formatted_data

def create_and_push_dataset(conversations: Dict[int, Dict[str, Any]], 
                          repo_id: str,
                          backup_path: str = "backup_conversations.jsonl"):
    """
    Create HuggingFace dataset and save local backup.
    
    Args:
        conversations: Processed conversations
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        backup_path: Path to save local backup JSONL file
    """
    # Convert to HF format
    formatted_data = conversations_to_hf_format(conversations)
    
    # Save local backup
    with open(backup_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Local backup saved to {backup_path}")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train/validation sets (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'validation': dataset['test']
    })
    
    # Push to HuggingFace
    dataset_dict.push_to_hub(repo_id)
    print(f"Dataset pushed to HuggingFace: {repo_id}")
    
    return dataset_dict