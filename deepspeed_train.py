import os
import json
import deepspeed
import torch
import transformers
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from tools.prepare_data import prepare_dialogue_dataset, prepare_sft_dataset
from codecarbon import EmissionsTracker

"""
This script is used to fine-tune the Gemma 2 2B IT model on the synthetic data.

To run the script, use the following command:
deepspeed --num_gpus=4 deepspeed_train.py

deepspeed --num_gpus=1 deepspeed_train.py
torchrun --nproc_per_node=1 deepspeed_train.py

Requirements:
    pip install wandb codecarbon
"""

def load_config():
    """Load configuration from JSON file"""
    with open('train_config.json', 'r') as f:
        return json.load(f)

def check_cuda():
    print("CUDA Environment:")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    print(f"- CUDA version: {torch.version.cuda}")
    print(f"- Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"- GPU model: {torch.cuda.get_device_name(0)}")
        print(f"- GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def get_deepspeed_config(config):
    """Return DeepSpeed configuration dictionary"""
    world_size = torch.cuda.device_count()
    print(f"Training configuration:")
    print(f"- Number of GPUs: {world_size}")
    return config['deepspeed']

def prepare_training_data(dataset, tokenizer, max_length):
    """Convert dataset to format expected by model."""
    
    def format_conversation(example):
        conversation = ""
        for msg in example['messages']:
            role = msg['role']
            content = msg['content']
            conversation += f"<|{role}|>\n{content}\n"
        conversation += "<|end|>"
        
        tokenized = tokenizer(
            conversation,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    formatted_dataset = dataset.map(
        format_conversation,
        remove_columns=dataset.column_names,
        desc="Converting conversations to tokens",
        num_proc=1,
    )
    
    return formatted_dataset

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Initialize distributed before anything else
        deepspeed.init_distributed()
        
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        check_cuda()
        
        # Initialize monitoring tools based on config flags
        wandb_run = None
        tracker = None
        
        if config['monitoring']['use_wandb']:
            # Initialize wandb
            wandb_run = wandb.init(
                project=config['monitoring']['wandb']['project'],
                name=config['monitoring']['wandb']['name'],
                config=config['monitoring']['wandb']['config']
            )
        
        if config['monitoring']['use_codecarbon']:
            # Create emissions directory and initialize tracker
            os.makedirs(config['monitoring']['emissions']['output_dir'], exist_ok=True)
            tracker = EmissionsTracker(
                project_name=config['monitoring']['wandb']['project'],
                output_dir=config['monitoring']['emissions']['output_dir'],
                log_level=config['monitoring']['emissions']['log_level']
            )
            tracker.start()
        
        # Load and prepare dataset
        chatml_data = prepare_dialogue_dataset()
        raw_dataset = prepare_sft_dataset(chatml_data)
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

        # Convert dataset
        dataset = prepare_training_data(raw_dataset, tokenizer, config['model']['max_length'])
        
        # Get DeepSpeed config
        ds_config = get_deepspeed_config(config)

        # Training arguments
        training_args = TrainingArguments(
            **config['training'],
            deepspeed=ds_config,
            # Only enable wandb reporting if it's enabled in config
            report_to="wandb" if config['monitoring']['use_wandb'] else "none"
        )

        # Print batch size info
        effective_batch_size = training_args.per_device_train_batch_size * \
                              training_args.gradient_accumulation_steps * \
                              torch.cuda.device_count()
        print(f"- Per device batch size: {training_args.per_device_train_batch_size}")
        print(f"- Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print(f"- Total batch size: {effective_batch_size}")

        # Create optimizer before DeepSpeed initialization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.0,  # Can be configured in train_config.json if needed
            betas=(0.9, 0.999)
        )

        # Initialize DeepSpeed before Trainer
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=training_args,
            model=model,
            optimizer=optimizer,  # Pass the optimizer here
            config=ds_config
        )

        # Initialize trainer with DeepSpeed engine
        trainer = Trainer(
            model=model_engine,
            args=training_args,
            train_dataset=dataset,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                return_tensors="pt",
            ),
        )
        
        try:
            trainer.train()
        finally:
            # Cleanup monitoring tools
            if tracker is not None:
                emissions = tracker.stop()
                print(f"Total emissions: {emissions} kg CO2eq")
                if wandb_run is not None:
                    wandb.log({"total_emissions_kg": emissions})
            
            if wandb_run is not None:
                wandb_run.finish()
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        if wandb.run is not None:
            wandb.run.finish(exit_code=1)
        raise
        
    finally:
        # Cleanup distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 