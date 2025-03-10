from typing import List, Dict, Optional, Union, Tuple, Any
import requests
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import pandas as pd
import time

class TokenLimitError(Exception):
    """Raised when token count exceeds model limits"""
    pass

def count_tokens(text: str, model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> int:
    """Count tokens in text using the model's tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

def truncate_context(
    messages: List[Dict],
    max_tokens: int = 3000,  # Leave room for completion
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
) -> List[Dict]:
    """
    Truncate context messages to fit within token limit.
    Keeps most recent messages.
    
    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens for context
        model_name: Model name for tokenizer
        
    Returns:
        Truncated list of messages
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Start from most recent messages
    truncated = []
    current_tokens = 0
    
    for msg in reversed(messages):
        # Count tokens in this message
        msg_tokens = len(tokenizer.encode(msg['text']))
        
        # Check if adding this message would exceed limit
        if current_tokens + msg_tokens > max_tokens:
            break
            
        truncated.insert(0, msg)
        current_tokens += msg_tokens
    
    return truncated

def prepare_api_request(
    messages: List[Dict],
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    top_p: float = 0.95
) -> Dict:
    """
    Prepare request payload for vLLM API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Name of the model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
    
    Returns:
        Dictionary containing the API request payload
    """
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

def call_vllm_api(
    payload: Dict,
    api_url: str = "http://0.0.0.0:8000/v1/chat/completions",
    headers: Optional[Dict] = None
) -> Dict:
    """
    Call vLLM API with the given payload.
    
    Args:
        payload: API request payload
        api_url: URL of the vLLM API
        headers: Optional request headers
    
    Returns:
        API response as dictionary
    """
    headers = headers or {"Content-Type": "application/json"}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(f"Response content: {response.text if 'response' in locals() else 'No response'}")
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to parse API response: {e}")
        print(f"Response content: {response.text}")
        raise

def format_conversation_for_generation(
    conversation: Dict,
    text_field: str = 'text'
) -> List[Dict]:
    """
    Format conversation messages for the LLM API.
    
    Args:
        conversation: Conversation dictionary from the dataset
        text_field: Field containing the message text
    
    Returns:
        List of formatted messages for the API
    """
    formatted_messages = []
    for msg in conversation['messages']:
        formatted_messages.append({
            "role": "user" if len(formatted_messages) % 2 == 0 else "assistant",
            "content": msg[text_field]
        })
    return formatted_messages

def parse_conversation_turns(
    messages: List[Dict],
    window_size: int = 3
) -> List[Tuple[List[Dict], Dict]]:
    """
    Parse conversation into context-response pairs.
    
    Args:
        messages: List of message dictionaries with 'text', 'post_number', 'poster_id'
        window_size: Number of messages to use as context
        
    Returns:
        List of (context, response) tuples
    """
    turns = []
    
    # Create sliding windows of messages
    for i in range(len(messages) - 1):
        # Get context window
        start_idx = max(0, i - window_size + 1)
        context = messages[start_idx:i + 1]
        response = messages[i + 1]
        
        # Only include if we have at least one context message
        if context:
            turns.append((context, response))
    
    return turns

def format_as_instruction_data(
    conversation_turns: List[Tuple[List[Dict], Dict]],
    conversation_id: int
) -> List[Dict]:
    """
    Format conversation turns as instruction data.
    
    Args:
        conversation_turns: List of (context, response) tuples
        conversation_id: ID of the conversation
        
    Returns:
        List of instruction examples
    """
    instruction_data = []
    
    for turn_num, (context, response) in enumerate(conversation_turns):
        # Format context messages
        formatted_context = []
        for msg in context:
            formatted_context.append({
                "role": "user" if len(formatted_context) % 2 == 0 else "assistant",
                "content": msg['text']
            })
        
        # Create instruction example
        example = {
            'conversation_id': conversation_id,
            'turn': turn_num,
            'prompt': formatted_context,
            'completion': [{
                "role": "assistant" if len(formatted_context) % 2 == 0 else "user",
                "content": response['text']
            }]
        }
        
        instruction_data.append(example)
    
    return instruction_data


def parse_generated_dialogue(response_text: str) -> List[Dict]:
    """
    Parse the generated dialogue text into message format.
    
    Args:
        response_text: Generated text from the model
        
    Returns:
        List of message dictionaries
    """
    messages = []
    post_number = 1
    
    # Split into lines and process each line
    for line in response_text.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        # Extract person and message
        try:
            person, text = line.split(':', 1)
            person = person.strip()
            text = text.strip()
            
            # Extract person number (assuming format "Person X")
            person_id = int(person.split()[-1])
            
            messages.append({
                'post_number': post_number,
                'poster_id': person_id,
                'text': text
            })
            post_number += 1
        except:
            continue
    
    return messages

def generate_synthetic_dialogue(
    conversation: Dict[str, List[Dict]],
    generation_config: Dict,
    context_size: int = 3,
    system_prompt: Optional[str] = None,
    api_url: str = "http://0.0.0.0:8000/v1/chat/completions",
    max_context_tokens: int = 3000
) -> List[Dict]:
    """
    Generate a synthetic dialogue with token limit handling.
    Uses the first messages of the conversation as context.
    Context is inserted into system prompt where {context} appears.
    
    Args:
        conversation: Dictionary with conversation_id and list of messages
        generation_config: Model configuration
        context_size: Number of messages to use as context
        system_prompt: System prompt with {context} placeholder
        api_url: vLLM API URL
        max_context_tokens: Maximum tokens for context
        
    Returns:
        List of generated messages
    """
    # Get context from start of conversation
    context_messages = conversation['messages'][:context_size]
    
    # Format context as text
    context_text = "\n".join(
        f"Person {msg['poster_id']}: {msg['text']}"
        for msg in context_messages
    )
    
    try:
        if system_prompt is None:
            raise ValueError("System prompt is required and must contain {context} placeholder")
            
        # Insert context into system prompt
        system_message = {
            "role": "system",
            "content": system_prompt.format(context=context_text)
        }
        
        # Create messages array with just the system prompt
        messages = [system_message]
        
        # Verify token count
        total_tokens = count_tokens(
            system_message["content"],
            model_name=generation_config["model"]
        )
        
        if total_tokens > max_context_tokens:
            raise TokenLimitError(
                f"Prompt exceeds token limit ({total_tokens} > {max_context_tokens})"
            )
        
        # Generate response
        payload = prepare_api_request(
            messages=messages,
            model=generation_config["model"],
            temperature=generation_config.get("temperature", 0.8),
            max_tokens=min(4096 - max_context_tokens, 1000),  # Adjust completion length
            top_p=generation_config.get("top_p", 0.95)
        )
        
        response = call_vllm_api(payload, api_url)
        
        if 'choices' not in response:
            print("Unexpected API response format:")
            print(json.dumps(response, indent=2))
            raise KeyError("API response missing 'choices' field")
        
        # Parse generated dialogue
        generated_messages = parse_generated_dialogue(
            response['choices'][0]['message']['content']
        )
        
        return generated_messages
        
    except TokenLimitError as e:
        print(f"Token limit error: {e}")
        if context_size > 1:
            print("Trying with smaller context...")
            # Retry with smaller context
            return generate_synthetic_dialogue(
                conversation=conversation,
                generation_config=generation_config,
                context_size=context_size - 1,
                system_prompt=system_prompt,
                api_url=api_url,
                max_context_tokens=max_context_tokens
            )
        else:
            raise
    except Exception as e:
        print(f"Error generating dialogue: {e}")
        if 'messages' in locals():
            print("Messages used:")
            print(json.dumps(messages, indent=2))
        raise 

def generate_dialogue_from_prompt(prompt: str, generation_config: dict) -> str:
    """Generate dialogue using vLLM API in OpenAI format."""
    try:
        # Prepare request in OpenAI chat format
        request_data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": generation_config["temperature"],
            "max_tokens": generation_config["max_tokens"],
            "top_p": generation_config["top_p"],
            "model": generation_config["model"]
        }
        
        response = requests.post(
            generation_config["api_config"]["vllm_url"],
            headers=generation_config["api_config"]["headers"],
            json=request_data,
            timeout=generation_config["request_timeout"]
        )
        response.raise_for_status()
        
        # Extract response in OpenAI format
        result = response.json()
        generated_text = result['choices'][0]['message']['content']
        return generated_text.strip()
        
    except Exception as e:
        print(f"Error in dialogue generation: {str(e)}")
        return None

def parse_generated_dialogue_to_messages(generated_dialogue: str, base_poster_ids: Dict[str, int] = {"Person 1": 1, "Person 2": 2}) -> List[Dict[str, Any]]:
    """Parse generated dialogue text into structured message format."""
    messages = []
    post_number = 1
    
    # Split into lines and process each message
    lines = [line.strip() for line in generated_dialogue.split('\n') if line.strip()]
    
    for line in lines:
        # Skip lines that don't match the Person X: format
        if not line.startswith(('Person 1:', 'Person 2:')):
            continue
            
        try:
            # Split into speaker and message
            speaker, text = line.split(':', 1)
            speaker = speaker.strip()
            text = text.strip()
            
            # Get poster_id from mapping
            poster_id = base_poster_ids.get(speaker)
            
            # Validate values before creating message
            if poster_id is None or pd.isna(poster_id):
                print(f"Warning: Invalid poster_id in line: {line}")
                continue
                
            if not text:
                print(f"Warning: Empty message text in line: {line}")
                continue
            
            # Create message dictionary
            message = {
                'post_number': post_number,
                'poster_id': int(poster_id),  # Ensure integer
                'text': str(text)  # Ensure string
            }
            
            messages.append(message)
            post_number += 1
            
        except Exception as e:
            print(f"Warning: Error parsing line '{line}': {e}")
            continue
    
    return messages

def generate_dataset(
    conversations: Dict[int, List[Dict]],
    generation_config: Dict[str, Any],
    prompt_template: str,
    is_conversation: bool = True,
    max_conversations: Optional[int] = None,
    min_messages: int = 2,
) -> Dict[str, Dict]:
    """
    Generate a dataset using the vLLM API. Can generate either conversations or single responses.
    
    Args:
        conversations: Dictionary mapping conversation_id to list of messages
        generation_config: Model generation configuration
        prompt_template: Template string with {context} placeholder
        is_conversation: If True, parses output as a conversation. If False, keeps raw output.
        max_conversations: Maximum number of conversations to process (None for all)
        min_messages: Minimum number of messages required in source conversation
        
    Returns:
        Dictionary with structure:
        {
            "conversation_id": {
                "original_messages": [...],
                "generated_output": str,
                "parsed_messages": [...],  # Only included if is_conversation=True
                "metadata": {
                    "conversation_id": int,
                    "num_messages": int,
                    "num_generated_messages": int,  # Only included if is_conversation=True
                }
            }
        }
    """
    generated_dataset = {}
    
    # Limit number of conversations if specified
    conv_items = list(conversations.items())
    if max_conversations:
        conv_items = conv_items[:max_conversations]
    
    # Process each conversation
    for conv_id, messages in tqdm(conv_items, desc="Generating dataset"):
        # Skip conversations with too few messages
        if len(messages) < min_messages:
            continue
            
        try:
            # Format context using first message
            context = f"{messages[0]['poster_id']}: {messages[0]['text']}"
            
            # Generate new content
            generated = generate_dialogue_from_prompt(
                prompt=prompt_template.format(context=context),
                generation_config=generation_config
            )
            
            if generated:
                # Initialize the dataset entry
                dataset_entry = {
                    "original_messages": messages,
                    "generated_output": generated,
                    "metadata": {
                        "conversation_id": conv_id,
                        "num_messages": len(messages),
                    }
                }
                
                # If this is a conversation, parse the messages
                if is_conversation:
                    parsed_messages = parse_generated_dialogue_to_messages(generated)
                    dataset_entry["parsed_messages"] = parsed_messages
                    dataset_entry["metadata"]["num_generated_messages"] = len(parsed_messages)
                
                generated_dataset[str(conv_id)] = dataset_entry
                
        except Exception as e:
            print(f"Error processing conversation {conv_id}: {e}")
            continue
            
    print(f"Successfully generated {len(generated_dataset)} outputs")
    return generated_dataset 

def create_analysis_dataset(
    generated_dataset: Dict[str, Dict],
    model_name: str
) -> List[Dict]:
    """
    Create a structured dataset for analysis from generated conversations.
    
    Args:
        generated_dataset: Output from generate_dataset()
        model_name: Name of the model used for generation
        
    Returns:
        List of dictionaries with structure:
        [
            {
                'model': str,
                'conversation_id': str,
                'turn_number': int,
                'original_message': {
                    'post_number': int,
                    'poster_id': int,
                    'text': str
                },
                'generated_message': {
                    'post_number': int,
                    'poster_id': int,
                    'text': str
                }
            },
            ...
        ]
    """
    analysis_dataset = []
    
    for conv_id, data in generated_dataset.items():
        # Get original and generated messages
        original_messages = data['original_messages']
        generated_messages = data['parsed_messages']
        
        # Determine max length for turn pairing
        max_turns = min(len(original_messages), len(generated_messages))
        
        # Create paired turns
        for turn in range(max_turns):
            turn_data = {
                'model': model_name,
                'conversation_id': conv_id,
                'turn_number': turn + 1,
                'original_message': original_messages[turn],
                'generated_message': generated_messages[turn]
            }
            analysis_dataset.append(turn_data)
    
    return analysis_dataset 

def create_hf_dataset(analysis_dataset, split_name="train", add_metadata=True):
    """
    Create a HuggingFace dataset from the analysis dataset.
    
    Args:
        analysis_dataset: List of dictionaries with analysis data
        split_name: Name of the split to create
        add_metadata: Whether to include metadata in the dataset
        
    Returns:
        HuggingFace Dataset object
    """
    from datasets import Dataset
    
    # Convert the analysis dataset to a format suitable for HuggingFace
    hf_data = []
    
    for item in analysis_dataset:
        # Extract the first message from each conversation for metadata
        first_orig_msg = item['orig_messages'][0] if item['orig_messages'] else {}
        first_synth_msg = item['synthetic_messages'][0] if item['synthetic_messages'] else {}
        
        # Create a dataset entry
        entry = {
            'conversation_id': item['conversation_id'],
            'messages': item['synthetic_messages'],
            'model': item['model'],
            'orig_message_count': item['orig_message_count'],
            'synthetic_message_count': item['synthetic_message_count'],
            'message_count_diff': item['message_count_diff'],
            'orig_total_length': item['orig_total_length'],
            'synthetic_total_length': item['synthetic_total_length'],
            'orig_total_tokens': item['orig_total_tokens'],
            'synthetic_total_tokens': item['synthetic_total_tokens'],
            'context_msg_used': item.get('context_msg_used', 0),
            'context_msg_available': item.get('context_msg_available', 0),
            'context_tokens_used': item.get('context_tokens_used', 0),
            'context_tokens_available': item.get('context_tokens_available', 0)
        }
        
        # Add metadata if requested
        if add_metadata and 'metadata' in item:
            entry['metadata'] = item['metadata']
        
        hf_data.append(entry)
    
    # Create the dataset
    return Dataset.from_list(hf_data) 