import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class TagType(Enum):
    QUOTE = "quote"
    BBCODE = "bbcode"
    HTML = "html"

@dataclass
class Tag:
    type: TagType
    start: int
    end: int
    content_start: int
    content_end: int
    attributes: Dict[str, str] = None
    nested_quotes: List['Tag'] = None

def find_all_quotes(text: str, start_pos: int = 0) -> List[Tag]:
    """
    Recursively find all quotes in text, including nested quotes.
    Returns a list of Tag objects with proper nesting structure.
    """
    quotes = []
    i = start_pos
    
    while i < len(text):
        # Look for either HTML or BBCode quote start
        html_match = re.search(r'<QUOTE([^>]*)>', text[i:], re.IGNORECASE)
        bbcode_match = re.search(r'\[quote(?:=.*?)?\]', text[i:], re.IGNORECASE)
        
        if not html_match and not bbcode_match:
            break
            
        # Determine which quote type comes first
        html_start = html_match.start() + i if html_match else float('inf')
        bbcode_start = bbcode_match.start() + i if bbcode_match else float('inf')
        
        if html_start < bbcode_start:
            # Handle HTML quote
            start = html_start
            attrs_str = html_match.group(1)
            quote_len = html_match.end()
            end_pattern = r'</QUOTE>'
        else:
            # Handle BBCode quote
            start = bbcode_start
            attrs_str = bbcode_match.group(0)[6:-1] if '=' in bbcode_match.group(0) else ''
            quote_len = bbcode_match.end()
            end_pattern = r'\[/quote\]'
        
        # Parse attributes
        attrs = {}
        for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
            attrs[attr_match.group(1)] = attr_match.group(2)
        
        # Find content boundaries
        content_start = start + quote_len
        
        # Find matching end tag, considering nesting
        stack = 1
        pos = content_start
        while stack > 0 and pos < len(text):
            start_match = re.search(r'<QUOTE|\[quote', text[pos:], re.IGNORECASE)
            end_match = re.search(end_pattern, text[pos:], re.IGNORECASE)
            
            if not end_match:
                break
                
            if not start_match or (end_match and end_match.start() < start_match.start()):
                stack -= 1
                pos += end_match.end()
            else:
                stack += 1
                pos += start_match.end()
        
        if stack == 0:
            end = pos
            content_end = end - len(end_pattern)
            
            # Create quote tag
            quote = Tag(
                type=TagType.QUOTE,
                start=start,
                end=end,
                content_start=content_start,
                content_end=content_end,
                attributes=attrs,
                nested_quotes=[]
            )
            
            # Recursively find nested quotes
            nested = find_all_quotes(text[content_start:content_end], 0)
            if nested:
                # Adjust positions for nested quotes
                for nested_quote in nested:
                    nested_quote.start += content_start
                    nested_quote.end += content_start
                    nested_quote.content_start += content_start
                    nested_quote.content_end += content_start
                quote.nested_quotes = nested
            
            quotes.append(quote)
            i = end
        else:
            i = start + 1
            
    return quotes

def remove_quotes(text: str) -> str:
    """
    Remove all quotes from text using a systematic parsing approach.
    
    Args:
        text: Text containing potential quotes
        
    Returns:
        Text with all quotes removed
    """
    # Handle None or empty text
    if not text or pd.isna(text):
        return ''
        
    # First handle the wrapping <r> tags
    r_match = re.match(r'<r>(.*)</r>', text, re.DOTALL)
    if r_match:
        text = r_match.group(1)
    
    # Find all quotes including nested structure
    quotes = find_all_quotes(text)
    
    # Sort quotes by start position in reverse order (to maintain positions when removing)
    all_quotes = []
    for quote in quotes:
        all_quotes.append(quote)
        if quote.nested_quotes:
            all_quotes.extend(quote.nested_quotes)
    
    all_quotes.sort(key=lambda x: x.start, reverse=True)
    
    # Remove quotes from end to start to maintain positions
    result = text
    for quote in all_quotes:
        result = result[:quote.start] + result[quote.end:]
    
    return result.strip()

def normalize_conversation(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a conversation by cleaning text and removing quotes."""
    normalized_conv = {
        'metadata': conversation['metadata'].copy(),
        'posts': []
    }
    
    for post in conversation['posts']:
        normalized_post = post.copy()
        normalized_post['original_text'] = post['text']
        
        # Remove quotes first
        text = remove_quotes(post['text'])
        
        # Basic BBCode/HTML cleanup
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\[.*?\]', '', text)  # Remove BBCode tags
        
        # Fix common entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Remove multiple newlines and spaces
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        normalized_post['normalized_text'] = text.strip()
        normalized_conv['posts'].append(normalized_post)
    
    return normalized_conv