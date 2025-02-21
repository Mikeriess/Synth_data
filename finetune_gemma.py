import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, setup_chat_format
from transformers import TrainingArguments, AutoTokenizer, TextStreamer, GenerationConfig
import getpass

# Configuration settings
RANDOM_SEED = 42

MODEL_CONFIGURATION = dict(
    model_name="google/gemma-2b-it",  # Using Gemma 2B instruction-tuned model
    max_seq_length=2048,
    dtype=None,  # Auto detect dtype
    load_in_4bit=True,  # Use 4-bit quantization to reduce memory usage
    attn_implementation="flash_attention_2"
)

PEFT_CONFIGURATION = dict(
    r=16,  # Adapter rank
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    use_rslora=False,
    loftq_config=None,
    random_state=RANDOM_SEED,
)

FINETUNING_CONFIGURATION = dict(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
)

def get_hf_token():
    """Get Hugging Face token from user input"""
    token = getpass.getpass("Hugging Face Token: ")
    if not token:
        print("Not using a Hugging Face token.")
        return None
    return token

def prepare_model(token=None):
    """Load and prepare the model for fine-tuning"""
    model, tokenizer = FastLanguageModel.from_pretrained(
        **MODEL_CONFIGURATION, 
        token=token
    )
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
    model = FastLanguageModel.get_peft_model(model, **PEFT_CONFIGURATION)
    return model, tokenizer

def prepare_dataset(dataset_name="kobprof/skolegpt-instruct", n_samples=1000):
    """Load and prepare the dataset"""
    dataset = load_dataset(dataset_name, split="train")
    print(f"Number of samples in dataset: {len(dataset):,}")
    
    # Take a random subset
    dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(n_samples))
    
    # Convert to ChatML format
    def create_conversation(sample):
        return {
            "messages": [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["response"]}
            ]
        }
    
    return dataset.map(create_conversation, batched=False)

def setup_trainer(model, tokenizer, dataset):
    """Set up the SFT trainer"""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=MODEL_CONFIGURATION["max_seq_length"],
        dataset_num_proc=4,
        packing=True,
        args=TrainingArguments(
            optim="adamw_8bit",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=3,
            seed=RANDOM_SEED,
            output_dir="outputs",
            **FINETUNING_CONFIGURATION
        ),
    )

def print_gpu_stats():
    """Print GPU memory statistics"""
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(
        f"Using {gpu_stats.name} GPU with {max_memory:.2f} GB total memory, "
        f"of which {start_gpu_memory:.2f} GB is reserved."
    )
    return start_gpu_memory, max_memory

def main():
    # Get HF token
    token = get_hf_token()
    
    # Prepare model and dataset
    model, tokenizer = prepare_model(token)
    dataset = prepare_dataset()
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, dataset)
    
    # Print initial GPU stats
    start_memory, max_memory = print_gpu_stats()
    
    # Start training
    print("Starting fine-tuning...")
    trainer_stats = trainer.train()
    
    # Print final GPU stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_memory, 3)
    print(
        f"Final GPU memory usage: {used_memory:.2f} GB "
        f"(LoRA used {used_memory_for_lora:.2f} GB)"
    )
    
    # Save the model
    output_dir = "gemma-2b-it-finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main() 