from typing import List, Dict, Callable, Optional
from tqdm import tqdm
import random
from .generator_utils import prepare_api_request, call_vllm_api, format_conversation_for_generation

def generate_conversation_continuations(
    dataset: List[Dict],
    prompt_template: Callable[[List[Dict]], List[Dict]],
    generation_config: Dict,
    api_url: str = "http://0.0.0.0:8000/v1/chat/completions",
    num_samples: Optional[int] = None,
    random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Generate continuations for conversations using the vLLM API.
    
    Args:
        dataset: List of conversation dictionaries
        prompt_template: Function that takes conversation messages and returns formatted prompt
        generation_config: Configuration for the generation model
        api_url: URL of the vLLM API
        num_samples: Number of conversations to sample (if None, uses all)
        random_seed: Random seed for sampling
        
    Returns:
        List of generated conversation continuations
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    # Sample conversations if specified
    conversations = dataset
    if num_samples is not None:
        conversations = random.sample(dataset, num_samples)
    
    generated_data = []
    
    for conv in tqdm(conversations, desc="Generating continuations"):
        # Format conversation messages
        messages = format_conversation_for_generation(conv)
        
        # Apply custom prompt template
        prompt = prompt_template(messages)
        
        # Prepare and send API request
        payload = prepare_api_request(
            messages=prompt,
            model=generation_config["model"],
            temperature=generation_config.get("temperature", 0.8),
            max_tokens=generation_config.get("max_tokens", 4096),
            top_p=generation_config.get("top_p", 0.95)
        )
        
        response = call_vllm_api(payload, api_url)
        
        # Format the result
        generated_data.append({
            'conversation_id': conv['conversation_id'],
            'original_messages': conv['messages'],
            'prompt': prompt,
            'completion': response['choices'][0]['message'],
            'model': generation_config["model"]
        })
    
    return generated_data 