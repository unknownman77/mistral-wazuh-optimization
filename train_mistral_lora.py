# 1. Backup file lama (kalau ada yang penting)
mv train_mistral_lora.py train_mistral_lora_old.py

# 2. Create file baru dengan script yang fixed
cat > train_mistral_lora.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed training script untuk fine-tuning Mistral 7B dengan LoRA
"""

import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_NAME = os.environ.get('BASE_MODEL', 'mistralai/Mistral-7B-v0.1')
DATA_PATH = os.environ.get('DATA_PATH', 'data/test_100.jsonl')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs/mistral-lora-test-100')
HF_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def load_and_prepare_data():
    """Load and prepare dataset"""
    logger.info(f"Loading dataset from {DATA_PATH}")
    
    try:
        ds = load_dataset('json', data_files=DATA_PATH)['train']
        logger.info(f"Dataset loaded successfully: {len(ds)} examples")
        return ds
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def format_example(example, tokenizer):
    """Format example for training"""
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    
    # Create prompt template
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}<|endoftext|>"
    
    # Tokenize
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None
    )
    
    # Set labels (for causal LM, labels = input_ids)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

def setup_model_and_tokenizer():
    """Setup model and tokenizer with quantization"""
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        use_fast=True,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model: {MODEL_NAME}")
    
    # Quantization config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora(model):
    """Setup LoRA configuration"""
    logger.info("Setting up LoRA")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # rank
        lora_alpha=32,  # scaling parameter
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    logger.info("Starting Mistral 7B LoRA fine-tuning")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available!")
        return
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check HuggingFace token
    if not HF_TOKEN:
        logger.error("HUGGINGFACE_TOKEN environment variable not set!")
        return
    
    # Load data
    dataset = load_and_prepare_data()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Prepare dataset
    logger.info("Preparing dataset")
    tokenized_dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset.column_names,
        batched=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,
    )
    
    # Get GPU memory and adjust batch size accordingly
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory >= 40:  # A100 or better
        batch_size = 4
        grad_accum = 4
    elif gpu_memory >= 24:  # RTX 4090, RTX 3090
        batch_size = 2
        grad_accum = 8
    else:  # Lower VRAM GPUs
        batch_size = 1
        grad_accum = 16
    
    logger.info(f"Using batch_size={batch_size}, grad_accum={grad_accum}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=min(100, len(dataset) // 10),  # 10% of data or 100 steps
        learning_rate=2e-4,
        fp16=True,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=min(500, len(dataset) // 2),  # Save twice during training
        save_total_limit=2,
        evaluation_strategy="no",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",  # More memory efficient
        lr_scheduler_type="cosine",
        report_to=None,  # Disable wandb/tensorboard for now
        max_steps=-1,  # Train for full epochs
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed successfully!")
    
    # Print training summary
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get('train_loss', 'N/A')
        logger.info(f"Final training loss: {final_loss}")

if __name__ == "__main__":
    main()
EOF

# 3. Set permission
chmod +x train_mistral_lora.py

# 4. Verify file fixed
head -30 train_mistral_lora.py
