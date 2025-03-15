#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd


# In[2]:


# Load raw data
with open("data/forums.pkl", 'rb') as f:
    forums = pickle.load(f)


with open("data/topics.pkl", 'rb') as f:
    topics = pickle.load(f)


with open("data/posts.pkl", 'rb') as f:
    posts = pickle.load(f)


# In[3]:


forums.head()


# In[4]:


forum_ids = [1,2,16,15,13]
forums[['forum_id', 'forum_name', 'forum_desc', 'parent_id']]


# In[5]:


# subset
forums =forums.loc[forums['forum_id'].isin(forum_ids)][['forum_id', 'forum_name']]
forums


# In[6]:


topics.head()


# In[7]:


# Subset topics
topics = topics.loc[topics['forum_id'].isin(forum_ids)][['topic_id', 'topic_title', 'topic_poster', 'forum_id']]
topics


# In[8]:


posts.head()


# In[9]:


posts.columns


# In[10]:


posts = posts.loc[posts['topic_id'].isin(topics['topic_id'])][['topic_id', 'post_id', 'post_text', 'post_time', 'poster_id', 'post_subject']]
posts


# # Merge posts, topics, and forums

# In[11]:


n_topics = 30000
subforums = [1,2]


# In[12]:


# Get forum IDs that are subforums (have a parent_id != 0)
subforum_ids = forums['forum_id'].unique()

# Randomly sample n_topics topics from those subforums
sampled_topics = topics[topics['forum_id'].isin(subforum_ids)].sample(n=n_topics, random_state=42)
sampled_topics


# In[13]:


# Merge topics with their forum info
merged_df = sampled_topics.merge(
    forums[['forum_id', 'forum_name']], 
    on='forum_id',
    how='left'
)

# Merge with posts
merged_df = merged_df.merge(
    posts[['topic_id', 'post_id', 'post_text', 'post_time', 'poster_id', 'post_subject']], 
    on='topic_id',
    how='left'
)

# Sort by topic_id and post_time to maintain conversation flow
merged_df = merged_df.sort_values(['forum_id','topic_id', 'post_time'])


# In[14]:


import html
# Clean HTML entities from forum names
merged_df['forum_name'] = merged_df['forum_name'].apply(html.unescape)


# In[15]:


merged_df


# In[16]:


print(f"Shape of merged dataset: {merged_df.shape}")
print("\nSample of columns in merged dataset:")
print(merged_df[['topic_id', 'forum_name', 'topic_title', 'post_subject', 'post_text']].head())


# In[17]:


# Save the merged dataframe to pickle format
merged_df.to_pickle('LM_sample_30000.pkl')


# In[ ]:




