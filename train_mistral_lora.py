# Minimal training script menggunakan QLoRA + PEFT
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import torch

MODEL_NAME = os.environ.get('BASE_MODEL', 'mistralai/Mistral-7B-v0.1')
DATA_PATH = os.environ.get('DATA_PATH', 'data/sample_train.jsonl')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs/mistral-lora')

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

ds = load_dataset('json', data_files=DATA_PATH)['train']

# Format simple prompt-response pairs
def format_example(ex):
prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
target = ex['response']
full = prompt + target
return {'input_ids': tokenizer(full, truncation=True, max_length=2048).input_ids}

ds = ds.map(format_example, remove_columns=ds.column_names)

bnb_config = BitsAndBytesConfig(llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(
MODEL_NAME,
load_in_4bit=True,
device_map='auto',
quantization_config=bnb_config,
trust_remote_code=True
)

lora_config = LoraConfig(
trainer.save_model(OUTPUT_DIR)