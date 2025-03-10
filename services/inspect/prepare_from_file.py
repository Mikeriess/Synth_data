import json
import os
import sys
from pathlib import Path
import uuid

def prepare_inspection_data_from_file(input_file: str, output_file: str = "conversations.json"):
    """
    Prepare dataset for the inspection interface from a processed JSON file.
    
    Args:
        input_file: Path to the processed JSON file
        output_file: Path to save the output JSON file
    """
    # Ensure the output directory exists
    output_path = Path(output_file)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Load the processed data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format required by the inspection interface
    conversations = []
    
    for i, item in enumerate(data):
        # Check if the item already has the conversation format
        if 'orig_messages' in item and 'synthetic_messages' in item:
            # Use the existing conversation format
            conversation = {
                'conversation_id': item.get('conversation_id', f"conv_{i}_{uuid.uuid4().hex[:8]}"),
                'orig_messages': item['orig_messages'],
                'synthetic_messages': item['synthetic_messages']
            }
            
            # Copy context statistics if available
            context_fields = [
                'context_msg_used', 'context_msg_available',
                'context_tokens_used', 'context_tokens_available'
            ]
            
            for field in context_fields:
                if field in item:
                    conversation[field] = item[field]
        else:
            # Create a conversation ID if not present
            conversation_id = item.get('id', f"conv_{i}_{uuid.uuid4().hex[:8]}")
            
            # Extract text content
            text = item.get('text', '')
            
            # Create a synthetic version by splitting the text into parts
            words = text.split()
            mid_point = len(words) // 2
            
            orig_text = ' '.join(words[:mid_point]) if mid_point > 0 else text
            synth_text = ' '.join(words[mid_point:]) if mid_point < len(words) else "Generated response."
            
            # Create conversation structure
            conversation = {
                'conversation_id': conversation_id,
                'orig_messages': [
                    {
                        'user': '1',
                        'text': orig_text
                    }
                ],
                'synthetic_messages': [
                    {
                        'user': '2',
                        'text': synth_text
                    }
                ]
            }
        
        conversations.append(conversation)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(conversations)} conversations for inspection")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_from_file.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "conversations.json"
    prepare_inspection_data_from_file(input_file, output_file) 