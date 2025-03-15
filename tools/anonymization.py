import re
from typing import Dict

def anonymize_conversation(conversation: Dict) -> Dict:
    """
    Anonymize all posts in a conversation while maintaining structure
    Returns updated conversation dict with anonymized texts
    """
    anonymized_conv = {
        'metadata': conversation['metadata'],
        'posts': []
    }
    
    for post in conversation['posts']:
        text = post['normalized_text']
        
        # Replace URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Replace usernames in quotes
        text = re.sub(r'quote=\"([^\"]+)\"', 'quote="[PER]"', text)
        
        # Replace names/usernames
        text = re.sub(r'@\w+', '[PER]', text)
        text = re.sub(r'/\w+$', '/[PER]', text)  # Signatures
        
        # Create anonymized post with previous versions preserved
        anonymized_post = post.copy()
        anonymized_post['anonymized_text'] = text
        anonymized_conv['posts'].append(anonymized_post)
    
    return anonymized_conv 