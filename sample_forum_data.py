#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forum data retrieval script for preparing data to be used with filter_and_replace.py.
This script loads forum data, processes it, and saves it in a format ready for anonymization.
"""

import pickle
import pandas as pd
import html
import argparse
from pathlib import Path
import random


def load_data(forums_path, topics_path, posts_path):
    """
    Load raw data from pickle files.
    
    Args:
        forums_path: Path to forums pickle file
        topics_path: Path to topics pickle file
        posts_path: Path to posts pickle file
        
    Returns:
        Tuple of (forums, topics, posts) DataFrames
    """
    print(f"Loading forums data from {forums_path}...")
    with open(forums_path, 'rb') as f:
        forums = pickle.load(f)
    
    print(f"Loading topics data from {topics_path}...")
    with open(topics_path, 'rb') as f:
        topics = pickle.load(f)
    
    print(f"Loading posts data from {posts_path}...")
    with open(posts_path, 'rb') as f:
        posts = pickle.load(f)
    
    return forums, topics, posts


def process_data(forums, topics, posts, forum_ids=None, n_topics=None, random_seed=42):
    """
    Process and merge forum data.
    
    Args:
        forums: Forums DataFrame
        topics: Topics DataFrame
        posts: Posts DataFrame
        forum_ids: List of forum IDs to include (default: None for all)
        n_topics: Number of topics to sample (default: None for all)
        random_seed: Random seed for sampling
        
    Returns:
        Merged DataFrame with forum conversations
    """
    # Filter forums if forum_ids is provided
    if forum_ids:
        print(f"Filtering to include only forums with IDs: {forum_ids}")
        forums = forums.loc[forums['forum_id'].isin(forum_ids)][['forum_id', 'forum_name']]
    else:
        # Use all forums but keep only necessary columns
        forums = forums[['forum_id', 'forum_name']]
    
    print(f"Working with {len(forums)} forums")
    
    # Get forum IDs to use for filtering topics
    subforum_ids = forums['forum_id'].unique()
    
    # Filter topics to only include those from selected forums
    topics = topics.loc[topics['forum_id'].isin(subforum_ids)]
    topics = topics[['topic_id', 'topic_title', 'topic_poster', 'forum_id']]
    
    print(f"Found {len(topics)} topics in selected forums")
    
    # Sample topics if n_topics is provided
    if n_topics and n_topics < len(topics):
        print(f"Sampling {n_topics} random topics...")
        sampled_topics = topics.sample(n=n_topics, random_state=random_seed)
    else:
        sampled_topics = topics
        print(f"Using all {len(sampled_topics)} topics")
    
    # Filter posts to only include those from selected topics
    posts = posts.loc[posts['topic_id'].isin(sampled_topics['topic_id'])]
    posts = posts[['topic_id', 'post_id', 'post_text', 'post_time', 'poster_id', 'post_subject']]
    
    print(f"Found {len(posts)} posts in selected topics")
    
    # Merge topics with their forum info
    print("Merging topics with forum information...")
    merged_df = sampled_topics.merge(
        forums[['forum_id', 'forum_name']], 
        on='forum_id',
        how='left'
    )
    
    # Merge with posts
    print("Merging with posts...")
    merged_df = merged_df.merge(
        posts[['topic_id', 'post_id', 'post_text', 'post_time', 'poster_id', 'post_subject']], 
        on='topic_id',
        how='left'
    )
    
    # Sort by topic_id and post_time to maintain conversation flow
    merged_df = merged_df.sort_values(['forum_id', 'topic_id', 'post_time'])
    
    # Clean HTML entities from forum names
    merged_df['forum_name'] = merged_df['forum_name'].apply(html.unescape)
    
    # Add post_number column (required by filter_and_replace.py)
    merged_df['post_number'] = merged_df.groupby('topic_id').cumcount()
    
    # Add conversation_id column (using topic_id)
    merged_df['conversation_id'] = merged_df['topic_id']
    
    return merged_df


def main():
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(description="Process forum data for anonymization")
    parser.add_argument("--forums", default="data/forums.pkl", 
                        help="Path to forums pickle file (default: data/forums.pkl)")
    parser.add_argument("--topics", default="data/topics.pkl", 
                        help="Path to topics pickle file (default: data/topics.pkl)")
    parser.add_argument("--posts", default="data/posts.pkl", 
                        help="Path to posts pickle file (default: data/posts.pkl)")
    parser.add_argument("--output", "-o", default="data/LM_sample_30000.pkl", 
                        help="Output pickle file path (default: data/LM_sample_30000.pkl)")
    parser.add_argument("--forum-ids", "-f", type=int, nargs="+", default=[1, 2],
                        help="Forum IDs to include (default: 1 2)")
    parser.add_argument("--n-topics", "-n", type=int, default=30000,
                        help="Number of topics to sample (default: 30000, use 0 for all)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    
    args = parser.parse_args()
    
    # Convert n_topics=0 to None (use all topics)
    n_topics = args.n_topics if args.n_topics > 0 else None
    
    # Load data
    forums, topics, posts = load_data(args.forums, args.topics, args.posts)
    
    # Process data
    merged_df = process_data(
        forums, 
        topics, 
        posts, 
        forum_ids=args.forum_ids,
        n_topics=n_topics,
        random_seed=args.seed
    )
    
    # Print summary
    print(f"\nProcessed dataset shape: {merged_df.shape}")
    print("\nSample of columns in processed dataset:")
    print(merged_df[['conversation_id', 'topic_id', 'forum_name', 'topic_title', 'post_subject', 'post_text']].head())
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the merged dataframe to pickle format
    print(f"\nSaving processed data to {args.output}...")
    merged_df.to_pickle(args.output)
    print("Done!")
    
    # Print instructions for using with filter_and_replace.py
    print("\nTo anonymize this data with filter_and_replace.py, run:")
    print(f"python filter_and_replace.py --input {args.output} --output data/output/anonymized_data.pkl")


if __name__ == "__main__":
    main() 