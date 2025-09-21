#!/usr/bin/env python3
import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get('BASE_MODEL', 'EleutherAI/gpt-neo-1.3B')
DATA_PATH = os.environ.get('DATA_PATH', 'data/test_100.jsonl')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs/gpt-neo-lora-test')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def format_example(example, tokenizer):
    instruction = example.get('instruction', '')
    response = example.get('response', '')
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}<|endoftext|>"
    
    return tokenizer(
        prompt,
        truncation=True,
        max_length=512,  # Smaller for consistent batching
        padding='max_length',
        return_tensors=None
    )

def main():
    logger.info("Starting GPT-Neo fine-tuning")
    
    # Load data
    ds = load_dataset('json', data_files=DATA_PATH)['train']
    logger.info(f"Dataset: {len(ds)} examples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"], bias="none"
    )
    model = get_peft_model(model, lora_config)
    
    # Prepare data
    train_dataset = ds.map(lambda x: format_example(x, tokenizer), remove_columns=ds.column_names)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=2e-4, fp16=True,
        logging_steps=5, save_steps=50, eval_strategy="no"
    )
    
    # Trainer
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
