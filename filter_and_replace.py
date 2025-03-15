#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data filtering and replacement script for anonymizing forum conversations.
This script implements the anonymization pipeline described in the paper,
following the same workflow as process.py.
"""

import re
import json
import pickle
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import spacy
from tqdm import tqdm
import argparse
from datasets import Dataset

# Import the same utility functions used in process.py
try:
    from tools.conversation_utils import organize_conversations
    from tools.normalization import normalize_conversation
    from tools.anonymization import anonymize_conversation
    from tools.hf_dataset import prepare_hf_dataset, upload_to_hub
    print("Successfully imported utility functions from utils and tools packages")
except ImportError:
    print("Warning: Could not import utility functions from utils/tools packages. Using built-in functions.")
    
    # Define fallback functions if imports fail
    def organize_conversations(data):
        """
        Organize raw data into conversation structure.
        
        Args:
            data: Raw data from pickle file
            
        Returns:
            Dictionary of conversations
        """
        conversations = {}
        
        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            print("Detected pandas DataFrame. Converting to conversation structure...")
            
            # Check if the DataFrame has conversation_id column
            if 'conversation_id' in data.columns:
                # Group by conversation_id
                for conv_id, group in data.groupby('conversation_id'):
                    # Create conversation metadata
                    conversations[conv_id] = {
                        'metadata': {
                            'topic_title': group['topic_title'].iloc[0] if 'topic_title' in group.columns else f'Conversation {conv_id}',
                            'forum_name': group['forum_name'].iloc[0] if 'forum_name' in group.columns else 'Unknown',
                            'total_posts': len(group)
                        },
                        'posts': []
                    }
                    
                    # Add posts
                    for _, row in group.iterrows():
                        post = {
                            'post_id': row.get('post_id', None),
                            'poster_id': row.get('poster_id', None),
                            'post_number': row.get('post_number', 0),
                            'post_time': row.get('post_time', None),
                            'original_text': row.get('text', row.get('original_text', ''))
                        }
                        conversations[conv_id]['posts'].append(post)
                    
                    # Sort posts by post_number
                    conversations[conv_id]['posts'].sort(key=lambda x: x['post_number'])
                
                return conversations
            
            # If no conversation_id, try to find other grouping columns
            elif any(col in data.columns for col in ['topic_id', 'thread_id']):
                # Find the grouping column
                group_col = next(col for col in ['topic_id', 'thread_id'] if col in data.columns)
                print(f"Using '{group_col}' as conversation grouping column")
                
                # Group by the identified column
                for conv_id, group in data.groupby(group_col):
                    # Create conversation metadata
                    conversations[conv_id] = {
                        'metadata': {
                            'topic_title': group['topic_title'].iloc[0] if 'topic_title' in group.columns else f'Conversation {conv_id}',
                            'forum_name': group['forum_name'].iloc[0] if 'forum_name' in group.columns else 'Unknown',
                            'total_posts': len(group)
                        },
                        'posts': []
                    }
                    
                    # Add posts
                    for _, row in group.iterrows():
                        post = {
                            'post_id': row.get('post_id', None),
                            'poster_id': row.get('poster_id', None),
                            'post_number': row.get('post_position', row.get('post_number', 0)),
                            'post_time': row.get('post_time', None),
                            'original_text': row.get('post_text', row.get('text', row.get('original_text', '')))
                        }
                        conversations[conv_id]['posts'].append(post)
                    
                    # Sort posts by post_number
                    conversations[conv_id]['posts'].sort(key=lambda x: x['post_number'])
                
                return conversations
            
            # If no grouping column, treat each row as a separate conversation
            else:
                print("No conversation grouping column found. Treating each row as a separate conversation.")
                # Print column names to help debug
                print(f"Available columns: {data.columns.tolist()}")
                
                # Create a conversation for each row
                for i, (_, row) in enumerate(data.iterrows()):
                    text_field = next((col for col in ['text', 'original_text', 'content'] if col in row and pd.notna(row[col])), None)
                    
                    if text_field is None:
                        # If no text field found, use the first non-null string column
                        for col in data.columns:
                            if isinstance(row[col], str) and pd.notna(row[col]):
                                text_field = col
                                break
                    
                    if text_field is None:
                        # If still no text field, use string representation of the row
                        text = str(row)
                    else:
                        text = row[text_field]
                    
                    conversations[i] = {
                        'metadata': {
                            'topic_title': f'Item {i}',
                            'forum_name': 'Unknown',
                            'total_posts': 1
                        },
                        'posts': [{
                            'post_id': i,
                            'poster_id': row.get('user_id', 0),
                            'post_number': 0,
                            'post_time': None,
                            'original_text': text
                        }]
                    }
        
        # Handle list of dictionaries
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            print("Detected list of dictionaries. Converting to conversation structure...")
            
            # Check if the dictionaries have a common structure
            if 'topic_id' in data[0]:
                # Group by topic_id
                for item in data:
                    conv_id = item['topic_id']
                    if conv_id not in conversations:
                        conversations[conv_id] = {
                            'metadata': {
                                'topic_title': item.get('topic_title', f'Topic {conv_id}'),
                                'forum_name': item.get('forum_name', 'Unknown'),
                                'total_posts': 0
                            },
                            'posts': []
                        }
                    
                    # Add post
                    post = {
                        'post_id': item.get('post_id', None),
                        'poster_id': item.get('poster_id', None),
                        'post_number': item.get('post_number', len(conversations[conv_id]['posts'])),
                        'post_time': item.get('post_time', None),
                        'original_text': item.get('text', item.get('content', ''))
                    }
                    conversations[conv_id]['posts'].append(post)
                    conversations[conv_id]['metadata']['total_posts'] += 1
            
            # If no topic_id, treat each item as a separate conversation
            else:
                for i, item in enumerate(data):
                    text_field = next((field for field in ['text', 'content'] if field in item), None)
                    
                    if text_field is None:
                        # If no text field found, use string representation of the item
                        text = str(item)
                    else:
                        text = item[text_field]
                    
                    conversations[i] = {
                        'metadata': {
                            'topic_title': item.get('title', f'Item {i}'),
                            'forum_name': item.get('forum', 'Unknown'),
                            'total_posts': 1
                        },
                        'posts': [{
                            'post_id': i,
                            'poster_id': item.get('user_id', 0),
                            'post_number': 0,
                            'post_time': None,
                            'original_text': text
                        }]
                    }
        
        # Handle list of strings
        elif isinstance(data, list):
            print("Detected list of items. Converting to conversation structure...")
            
            # Treat each item as a separate conversation
            for i, item in enumerate(data):
                conversations[i] = {
                    'metadata': {
                        'topic_title': f'Item {i}',
                        'forum_name': 'Unknown',
                        'total_posts': 1
                    },
                    'posts': [{
                        'post_id': i,
                        'poster_id': 0,
                        'post_number': 0,
                        'post_time': None,
                        'original_text': str(item)
                    }]
                }
        
        return conversations
    
    def normalize_conversation(conversation):
        """Fallback normalize_conversation function"""
        normalized_conversation = conversation.copy()
        
        for post in normalized_conversation['posts']:
            # Basic normalization
            text = post['original_text']
            
            # Handle non-string text
            if not isinstance(text, str):
                text = str(text)
            
            # Remove BBCode quotes
            text = re.sub(r'\[quote(?:=[^\]]+)?\](.*?)\[/quote\]', '', text, flags=re.DOTALL)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Normalize quotes
            text = re.sub(r'["""]', '"', text)
            
            # Normalize apostrophes
            text = re.sub(r'[''`]', "'", text)
            
            # Normalize dashes
            text = re.sub(r'[–—]', '-', text)
            
            # Add normalized text to post
            post['normalized_text'] = text.strip()
        
        return normalized_conversation
    
    def anonymize_conversation(conversation):
        """Fallback anonymize_conversation function"""
        # Implementation as before...
        anonymized_conversation = conversation.copy()
        
        # Create anonymizer
        anonymizer = DataAnonymizer()
        
        for post in anonymized_conversation['posts']:
            # Get NER-anonymized text if available, otherwise use normalized text
            text = post.get('ner_anonymized_text', post.get('normalized_text', post.get('original_text', '')))
            
            # Detect entities and patterns
            entities = anonymizer.detect_entities(text)
            patterns = anonymizer.detect_patterns(text)
            
            # Replace entities and patterns
            anonymized_text = anonymizer.replace_entities(text, entities)
            anonymized_text = anonymizer.replace_patterns(anonymized_text, patterns)
            
            # Add anonymized text to post
            post['anonymized_text'] = anonymized_text
        
        return anonymized_conversation

# Define helper functions for data processing
def conversations_to_dataframe(conversations):
    """
    Convert conversations dictionary to a pandas DataFrame.
    
    Args:
        conversations: Dictionary of conversations
        
    Returns:
        DataFrame with one row per post, including conversation metadata
    """
    rows = []
    
    # Check if we have any conversations
    if not conversations:
        print("Warning: No conversations to convert to DataFrame")
        # Return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=[
            'conversation_id', 'post_number', 'post_id', 'poster_id', 
            'post_time', 'forum_name', 'topic_title', 'total_posts_in_conversation',
            'original_text', 'normalized_text', 'anonymized_text'
        ])
    
    for conv_id, conversation in conversations.items():
        metadata = conversation['metadata']
        
        for post in conversation['posts']:
            # Handle post_id and poster_id to ensure they're integers
            try:
                post_id = int(post['post_id']) if post['post_id'] is not None else None
            except (ValueError, TypeError):
                post_id = post['post_id']  # Keep original if can't convert to int
                
            try:
                poster_id = int(post['poster_id']) if post['poster_id'] is not None else None
            except (ValueError, TypeError):
                poster_id = post['poster_id']  # Keep original if can't convert to int
            
            # Create a unique row for each post with all metadata
            row = {
                'conversation_id': conv_id,
                'post_number': post.get('post_number', 0),
                'post_id': post_id,
                'poster_id': poster_id,
                'post_time': post.get('post_time', None),
                'forum_name': metadata['forum_name'],
                'topic_title': metadata['topic_title'],
                'total_posts_in_conversation': int(metadata['total_posts']),
                'original_text': post['original_text'],
                'normalized_text': post.get('normalized_text', post['original_text']),
                'anonymized_text': post.get('anonymized_text', post.get('normalized_text', post['original_text']))
            }
            rows.append(row)
            
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert post_id and poster_id columns to Int64 (pandas nullable integer type)
    if 'post_id' in df.columns:
        df['post_id'] = pd.to_numeric(df['post_id'], errors='coerce').astype('Int64')
    
    if 'poster_id' in df.columns:
        df['poster_id'] = pd.to_numeric(df['poster_id'], errors='coerce').astype('Int64')
    
    # Sort by conversation_id and post_number if both columns exist
    if not df.empty and 'conversation_id' in df.columns and 'post_number' in df.columns:
        df = df.sort_values(['conversation_id', 'post_number'])
    
    return df

def count_tokens(df, text_columns, conversation_id_col, tokenizer_name=None):
    """
    Count tokens in text columns of a DataFrame.
    
    Args:
        df: DataFrame with text columns
        text_columns: List of column names containing text
        conversation_id_col: Column name for conversation ID
        tokenizer_name: Name of HuggingFace tokenizer to use
        
    Returns:
        token_stats: Dictionary with token statistics for each text column
        conv_stats: Dictionary with conversation-level statistics
    """
    try:
        from transformers import AutoTokenizer
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Use google/gemma-2-27b-it as the default tokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
    except ImportError:
        print("Warning: transformers package not found. Using simple whitespace tokenization.")
        tokenizer = None
    
    token_stats = {}
    conv_stats = {}
    
    # Calculate token counts for each text column
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
            
        # Count tokens
        if tokenizer:
            # Use HuggingFace tokenizer
            token_counts = df[col].apply(
                lambda x: len(tokenizer.encode(str(x))) if pd.notna(x) else 0
            )
        else:
            # Simple whitespace tokenization
            token_counts = df[col].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        # Add token counts to DataFrame
        token_col = f"{col}_tokens"
        df[token_col] = token_counts
        
        # Calculate statistics
        token_stats[col] = {
            'total_tokens': token_counts.sum(),
            'mean_tokens': token_counts.mean(),
            'median_tokens': token_counts.median(),
            'min_tokens': token_counts.min(),
            'max_tokens': token_counts.max(),
            'std_tokens': token_counts.std()
        }
    
    # Calculate conversation-level statistics
    if conversation_id_col in df.columns:
        for col in text_columns:
            token_col = f"{col}_tokens"
            if token_col in df.columns:
                # Group by conversation and calculate statistics
                grouped = df.groupby(conversation_id_col)[token_col].agg(['sum', 'mean', 'median', 'min', 'max', 'count'])
                
                # Store in conv_stats
                conv_stats[col] = {
                    'total_conversations': len(grouped),
                    'mean_tokens_per_conversation': grouped['sum'].mean(),
                    'median_tokens_per_conversation': grouped['sum'].median(),
                    'min_tokens_per_conversation': grouped['sum'].min(),
                    'max_tokens_per_conversation': grouped['sum'].max(),
                    'mean_messages_per_conversation': grouped['count'].mean(),
                    'median_messages_per_conversation': grouped['count'].median(),
                    'min_messages_per_conversation': grouped['count'].min(),
                    'max_messages_per_conversation': grouped['count'].max()
                }
    
    return token_stats, conv_stats

def prepare_conversation_dataset(df, text_field, message_features=None, conversation_features=None, include_token_stats=False):
    """
    Prepare conversation dataset for HuggingFace.
    
    Args:
        df: DataFrame with conversation data
        text_field: Column name for the text field to use
        message_features: List of message-level features to include
        conversation_features: List of conversation-level features to include
        include_token_stats: Whether to include token statistics
        
    Returns:
        List of formatted conversations
    """
    if message_features is None:
        message_features = []
    
    if conversation_features is None:
        conversation_features = []
    
    formatted_conversations = []
    
    # Group by conversation_id
    if 'conversation_id' in df.columns:
        for conv_id, group in df.groupby('conversation_id'):
            # Sort by post_number
            group = group.sort_values('post_number')
            
            # Extract conversation-level features
            conv_data = {
                'conversation_id': conv_id,
                'messages': []
            }
            
            # Add conversation-level features
            for feature in conversation_features:
                if feature in group.columns:
                    # Convert pandas NA to None for HuggingFace compatibility
                    value = group[feature].iloc[0]
                    if pd.isna(value):
                        value = None
                    conv_data[feature] = value
            
            # Add token statistics if requested
            if include_token_stats and f"{text_field}_tokens" in group.columns:
                conv_data['total_tokens'] = group[f"{text_field}_tokens"].sum()
                conv_data['num_messages'] = len(group)
            
            # Add messages
            for _, row in group.iterrows():
                message = {
                    'text': row[text_field] if pd.notna(row[text_field]) else ""
                }
                
                # Add message-level features
                for feature in message_features:
                    if feature in row:
                        # Convert pandas NA to None for HuggingFace compatibility
                        value = row[feature]
                        if pd.isna(value):
                            value = None
                        message[feature] = value
                
                conv_data['messages'].append(message)
            
            formatted_conversations.append(conv_data)
    
    return formatted_conversations

def print_token_stats(token_stats, conv_stats):
    """
    Print token statistics.
    
    Args:
        token_stats: Dictionary with token statistics for each text column
        conv_stats: Dictionary with conversation-level statistics
    """
    print("\nToken Statistics:")
    
    for col, stats in token_stats.items():
        print(f"\n{col}:")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Mean tokens per message: {stats['mean_tokens']:.2f}")
        print(f"  Median tokens per message: {stats['median_tokens']:.2f}")
        print(f"  Min tokens per message: {stats['min_tokens']}")
        print(f"  Max tokens per message: {stats['max_tokens']}")
    
    print("\nConversation Statistics:")
    
    for col, stats in conv_stats.items():
        print(f"\n{col}:")
        print(f"  Total conversations: {stats['total_conversations']:,}")
        print(f"  Mean tokens per conversation: {stats['mean_tokens_per_conversation']:.2f}")
        print(f"  Median tokens per conversation: {stats['median_tokens_per_conversation']:.2f}")
        print(f"  Mean messages per conversation: {stats['mean_messages_per_conversation']:.2f}")
        print(f"  Median messages per conversation: {stats['median_messages_per_conversation']:.2f}")

def print_dataset_info(formatted_conversations):
    """
    Print dataset information.
    
    Args:
        formatted_conversations: List of formatted conversations
    """
    print("\nDataset Information:")
    print(f"  Total conversations: {len(formatted_conversations):,}")
    
    if formatted_conversations:
        total_messages = sum(len(conv['messages']) for conv in formatted_conversations)
        print(f"  Total messages: {total_messages:,}")
        print(f"  Average messages per conversation: {total_messages / len(formatted_conversations):.2f}")
        
        if 'total_tokens' in formatted_conversations[0]:
            total_tokens = sum(conv['total_tokens'] for conv in formatted_conversations)
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Average tokens per conversation: {total_tokens / len(formatted_conversations):.2f}")
            print(f"  Average tokens per message: {total_tokens / total_messages:.2f}")

def apply_ner_anonymization(conversation, nlp=None):
    """
    Apply NER-based anonymization using daCy's large transformer model.
    This identifies and anonymizes entities that are problematic under GDPR.
    
    Args:
        conversation: Normalized conversation
        nlp: Pre-loaded daCy model (optional)
        
    Returns:
        Conversation with NER-based anonymization applied
    """
    if nlp is None:
        try:
            import dacy
            # Load daCy large transformer model
            print("Loading daCy large transformer model...")
            nlp = dacy.load("da_dacy_large_trf-0.2.0")
            print("Model loaded successfully.")
        except (ImportError, OSError) as e:
            print(f"Warning: Could not load daCy model: {str(e)}")
            print("Skipping NER-based anonymization.")
            return conversation
    
    # Create a copy of the conversation
    ner_anonymized = conversation.copy()
    
    # Define entity types to anonymize (GDPR-sensitive entities)
    gdpr_entity_types = {
        "PER": "[PERSON]",      # Person names
        "LOC": "[LOCATION]",    # Locations that might identify someone
        "DATE": "[DATE]",       # Dates that might be personally identifiable
        "NORP": "[GROUP]",      # Nationalities, religious or political groups
        # Other entity types remain commented out to preserve technical content
        # "ORG": "[ORGANIZATION]",
        # "MISC": "[MISC]",
        # "GPE": "[LOCATION]",
        # "FAC": "[FACILITY]",
        # "EVENT": "[EVENT]",
        # "WORK_OF_ART": "[WORK]",
        # "LAW": "[LAW]",
        # "LANGUAGE": "[LANGUAGE]",
        # "TIME": "[TIME]",
        # "MONEY": "[MONEY]",
        # "QUANTITY": "[QUANTITY]",
        # "ORDINAL": "[ORDINAL]",
        # "CARDINAL": "[NUMBER]"
    }
    
    # Add signature detection regex
    signature_patterns = [
        r'(?:^|\s)\/([A-Z][a-z]+)(?:\s|$)',  # /Michael
        r'(?:^|\s)Mvh\.?\s+([A-Z][a-z]+)',   # Mvh. Michael
        r'(?:^|\s)Hilsen\s+([A-Z][a-z]+)',   # Hilsen Michael
        r'(?:^|\s)Regards\s+([A-Z][a-z]+)',  # Regards Michael
        r'(?:^|\s)Venlig hilsen\s+([A-Z][a-z]+)', # Venlig hilsen Michael
        r'(?:^|\s)Med venlig hilsen\s+([A-Z][a-z]+)', # Med venlig hilsen Michael
        r'(?:^|\s)Kh\.?\s+([A-Z][a-z]+)',    # Kh. Michael
        r'(?:^|\s)Vh\.?\s+([A-Z][a-z]+)',    # Vh. Michael
    ]
    
    # Process each post
    for post in ner_anonymized['posts']:
        # Get normalized text
        text = post.get('normalized_text', post.get('original_text', ''))
        
        if not text or not isinstance(text, str):
            continue
        
        # Process text with NER model
        try:
            # Handle long texts by splitting into chunks
            max_length = 512  # Maximum safe length for transformer models
            anonymized_text = text
            
            if len(text) > max_length:
                # Process in overlapping chunks
                chunks = []
                for i in range(0, len(text), max_length - 50):  # 50 token overlap
                    chunk = text[i:i + max_length]
                    chunks.append(chunk)
                
                # Process each chunk and collect entities
                all_entities = []
                for chunk in chunks:
                    doc = nlp(chunk)
                    chunk_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                                     for ent in doc.ents if ent.label_ in gdpr_entity_types]
                    
                    # Adjust entity positions based on chunk offset
                    chunk_offset = text.find(chunk)
                    if chunk_offset >= 0:  # Only if chunk is found in original text
                        adjusted_entities = [
                            (ent[0], ent[1], ent[2] + chunk_offset, ent[3] + chunk_offset)
                            for ent in chunk_entities
                        ]
                        all_entities.extend(adjusted_entities)
            else:
                # Process the entire text at once
                doc = nlp(text)
                all_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                               for ent in doc.ents if ent.label_ in gdpr_entity_types]
            
            # Sort entities by position (from end to start to avoid offset issues)
            all_entities.sort(key=lambda x: x[2], reverse=True)
            
            # Replace entities with placeholders
            for entity_text, entity_type, start_pos, end_pos in all_entities:
                replacement = gdpr_entity_types.get(entity_type, f"[{entity_type}]")
                anonymized_text = anonymized_text[:start_pos] + replacement + anonymized_text[end_pos:]
            
            # Additional step: Detect and anonymize signatures
            for pattern in signature_patterns:
                anonymized_text = re.sub(pattern, r' [PERSON]', anonymized_text)
            
            # Store the updated anonymized text
            post['ner_anonymized_text'] = anonymized_text
            
        except Exception as e:
            print(f"Error in NER processing: {str(e)}")
            # Keep original text if NER fails
            post['ner_anonymized_text'] = text
    
    return ner_anonymized

# Add this class definition before the anonymize_conversation function
class DataAnonymizer:
    """
    Class for anonymizing personal data in text.
    Detects and replaces entities and patterns that might contain personal information.
    """
    
    def __init__(self):
        # Define regex patterns for personal information
        self.patterns = {
            'email': (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            'phone': (r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', '[PHONE]'),
            'url': (r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*\??[-\w&=.]*', '[URL]'),
            'ip': (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP_ADDRESS]'),
            'ssn': (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]'),
            'credit_card': (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[CREDIT_CARD]'),
            'address': (r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|court|ct|drive|dr|way|parkway|pkwy)\b', '[ADDRESS]'),
            'zipcode': (r'\b\d{5}(?:-\d{4})?\b', '[ZIPCODE]')
        }
        
        # Load usernames from file
        self.usernames = set()
        try:
            with open('data/usernames.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    username = line.strip()
                    if username:  # Skip empty lines
                        self.usernames.add(username)
            print(f"Loaded {len(self.usernames)} usernames for anonymization")
        except FileNotFoundError:
            print("Warning: Username file 'data/usernames.txt' not found")
    
    def detect_patterns(self, text):
        """
        Detect patterns in text that might contain personal information.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (pattern_type, start_pos, end_pos) tuples
        """
        if not text or not isinstance(text, str):
            return []
        
        patterns_found = []
        
        # Check for regex patterns
        for pattern_type, (regex, _) in self.patterns.items():
            for match in re.finditer(regex, text, re.IGNORECASE):
                patterns_found.append((pattern_type, match.start(), match.end()))
        
        # Check for usernames
        if self.usernames:
            # Sort usernames by length (longest first) to avoid partial matches
            sorted_usernames = sorted(self.usernames, key=len, reverse=True)
            
            for username in sorted_usernames:
                # Find all occurrences of the username
                for match in re.finditer(r'\b' + re.escape(username) + r'\b', text, re.IGNORECASE):
                    patterns_found.append(('username', match.start(), match.end()))
        
        # Sort by position (from end to start to avoid offset issues)
        patterns_found.sort(key=lambda x: x[1], reverse=True)
        
        return patterns_found
    
    def detect_entities(self, text):
        """
        Detect entities in text that might contain personal information.
        This is a placeholder for more sophisticated entity detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of (entity_type, start_pos, end_pos) tuples
        """
        # This is a simple implementation
        # In a real system, you might use a more sophisticated NER model
        return []
    
    def replace_patterns(self, text, patterns_found):
        """
        Replace patterns with placeholders.
        
        Args:
            text: Original text
            patterns_found: List of (pattern_type, start_pos, end_pos) tuples
            
        Returns:
            Text with patterns replaced by placeholders
        """
        if not text or not isinstance(text, str) or not patterns_found:
            return text
        
        anonymized_text = text
        
        for pattern_type, start_pos, end_pos in patterns_found:
            if pattern_type == 'username':
                replacement = '[PERSON]'
            else:
                replacement = self.patterns.get(pattern_type, ('', '[REDACTED]'))[1]
            
            anonymized_text = anonymized_text[:start_pos] + replacement + anonymized_text[end_pos:]
        
        return anonymized_text
    
    def replace_entities(self, text, entities_found):
        """
        Replace entities with placeholders.
        
        Args:
            text: Original text
            entities_found: List of (entity_type, start_pos, end_pos) tuples
            
        Returns:
            Text with entities replaced by placeholders
        """
        if not text or not isinstance(text, str) or not entities_found:
            return text
        
        anonymized_text = text
        
        for entity_type, start_pos, end_pos in entities_found:
            replacement = f"[{entity_type.upper()}]"
            anonymized_text = anonymized_text[:start_pos] + replacement + anonymized_text[end_pos:]
        
        return anonymized_text

def main():
    """Main function to run the anonymization pipeline."""
    parser = argparse.ArgumentParser(description="Anonymize forum conversation data")
    parser.add_argument("--input", "-i", default="data/LM_sample_1000.pkl", 
                        help="Input pickle file path (default: data/LM_sample_1000.pkl)")
    parser.add_argument("--output", "-o", default="data/output/anonymized_data.pkl", 
                        help="Output pickle file path (default: data/output/anonymized_data.pkl)")
    parser.add_argument("--hf-dataset", help="Name for HuggingFace dataset (optional)")
    parser.add_argument("--private", action="store_true", help="Make HuggingFace dataset private")
    parser.add_argument("--debug", action="store_true", default=True, 
                        help="Print debug information (default: enabled)")
    parser.add_argument("--sample", "-s", type=int, default=0,
                        help="Number of random conversations to process (default: 0 for all, use positive number for sample)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        with open(args.input, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        print("Please make sure the file exists or specify a different input path with --input.")
        return
    
    # Debug: Print data type and sample
    if args.debug:
        print(f"Data type: {type(data)}")
        if isinstance(data, list):
            print(f"First item type: {type(data[0]) if data else 'empty list'}")
            if data and isinstance(data[0], dict):
                print(f"First item keys: {list(data[0].keys())}")
            elif data:
                print(f"First item: {data[0][:100] if isinstance(data[0], str) else data[0]}")
    
    print(f"Loaded data with {len(data)} items")
    
    # Step 1: Organize conversations
    print("Organizing conversations...")
    conversations = organize_conversations(data)
    print(f"Organized into {len(conversations)} conversations")
    
    # If no conversations, exit
    if len(conversations) == 0:
        print("Error: Could not organize data into conversations. Exiting.")
        return
    
    # Sample a subset of conversations if requested
    if args.sample > 0 and args.sample < len(conversations):
        print(f"Sampling {args.sample} random conversations...")
        # Get random sample of conversation IDs
        sampled_conv_ids = random.sample(list(conversations.keys()), args.sample)
        # Create new dictionary with only sampled conversations
        conversations = {conv_id: conversations[conv_id] for conv_id in sampled_conv_ids}
        print(f"Sampled {len(conversations)} conversations")
    
    # Step 2: Normalize conversations
    print("Normalizing conversations...")
    normalized_conversations = {
        conv_id: normalize_conversation(conv)
        for conv_id, conv in tqdm(conversations.items(), desc="Normalizing")
    }
    
    # Step 3: Apply NER-based anonymization
    print("Applying NER-based anonymization...")
    try:
        import dacy
        print("Loading daCy large transformer model (once for all conversations)...")
        nlp = dacy.load("da_dacy_large_trf-0.2.0")
        print("Model loaded successfully.")
        
        ner_anonymized_conversations = {
            conv_id: apply_ner_anonymization(conv, nlp=nlp)
            for conv_id, conv in tqdm(normalized_conversations.items(), desc="NER Anonymization")
        }
    except (ImportError, OSError) as e:
        print(f"Warning: Could not load daCy model: {str(e)}")
        print("Skipping NER-based anonymization step.")
        # Skip NER step if model can't be loaded
        ner_anonymized_conversations = normalized_conversations
    
    # Step 4: Apply pattern-based anonymization
    print("Applying pattern-based anonymization...")
    anonymized_conversations = {
        conv_id: anonymize_conversation(conv)
        for conv_id, conv in tqdm(ner_anonymized_conversations.items(), desc="Pattern Anonymization")
    }
    
    # Step 5: Convert to DataFrame
    print("Converting to DataFrame...")
    conversations_df = conversations_to_dataframe(anonymized_conversations)
    
    # Step 6: Count tokens
    print("Counting tokens...")
    text_columns = ['original_text', 'normalized_text', 'anonymized_text']
    token_stats, conv_stats = count_tokens(
        conversations_df,
        text_columns=text_columns,
        conversation_id_col='conversation_id',
        tokenizer_name="google/gemma-2-27b-it"
    )
    
    # Print token statistics
    print_token_stats(token_stats, conv_stats)
    
    # Step 7: Prepare conversation dataset
    print("Preparing conversation dataset...")
    formatted_conversations = prepare_conversation_dataset(
        conversations_df,
        text_field='anonymized_text',
        message_features=['post_number', 'poster_id'],
        conversation_features=['forum_name', 'total_posts_in_conversation'],
        include_token_stats=True
    )
    
    # Print dataset information
    print_dataset_info(formatted_conversations)
    
    # Step 8: Convert to HuggingFace Dataset
    print("Converting to HuggingFace Dataset...")
    try:
        hf_dataset = Dataset.from_list(formatted_conversations)
    except Exception as e:
        print(f"Warning: Error creating HuggingFace Dataset: {str(e)}")
        print("Attempting to fix NA values...")
        
        # Convert all NA values to None in the formatted conversations
        def fix_na_values(obj):
            if isinstance(obj, dict):
                return {k: fix_na_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_na_values(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        fixed_conversations = fix_na_values(formatted_conversations)
        hf_dataset = Dataset.from_list(fixed_conversations)
    
    # Save processed data
    print(f"Saving processed data to {args.output}...")
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(anonymized_conversations, f)
    
    # Save DataFrame
    df_output = str(output_path).replace('.pkl', '_df.pkl')
    print(f"Saving DataFrame to {df_output}...")
    conversations_df.to_pickle(df_output)
    
    # Upload to HuggingFace Hub if requested
    if args.hf_dataset:
        print(f"Uploading to HuggingFace Hub as {args.hf_dataset}...")
        hf_dataset.push_to_hub(
            args.hf_dataset,
            private=args.private,
            token=True
        )
    
    print("Done!")

if __name__ == "__main__":
    main() 