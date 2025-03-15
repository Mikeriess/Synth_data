from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from unsloth import FastLanguageModel

def test_finetuned_model(model_path="gemma-2b-it-finetuned"):
    # Load the fine-tuned model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Setup generation config
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    
    # Test prompt
    messages = [
        {"role": "user", "content": "What have you learned from your fine-tuning?"}
    ]
    
    # Generate response
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
        generation_config=generation_config,
    )

if __name__ == "__main__":
    test_finetuned_model() 