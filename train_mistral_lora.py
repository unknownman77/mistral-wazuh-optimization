import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # load model
    model_name = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # tentukan dtype otomatis: bf16 kalau tersedia, else fp16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
        logger.info("Using bf16 for training")
    else:
        dtype = torch.float16
        logger.info("Using fp16 for training")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=dtype
    )

    # apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # load dataset
    dataset = load_dataset(
        "json",
        data_files={"train": "data/logdata_wazuh_smart.jsonl"},
        split="train"
    )
    dataset = dataset.map(lambda x: {"text": x["instruction"] + "\n" + x["response"]})

    # training arguments
    training_args = TrainingArguments(
        output_dir="./mistral-lora-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args
    )

    # start training
    logger.info("Starting Mistral-7B LoRA fine-tuning...")
    trainer.train()
    logger.info("Training finished, thanks UnknownMan!")

if __name__ == "__main__":
    main()
