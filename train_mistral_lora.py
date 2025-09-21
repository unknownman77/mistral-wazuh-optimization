import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Base model: Mistral-7B
    model_name = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loading model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # penting agar tidak error padding

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # modul utama Mistral
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Load dataset JSONL
    dataset = load_dataset(
        "json",
        data_files={"train": "data/logdata_wazuh_smart.jsonl"},
        split="train"
    )

    # Pastikan ada field text
    if "text" not in dataset.column_names:
        logger.info("Mapping dataset field 'message' -> 'text'")
        dataset = dataset.map(lambda x: {"text": x["message"]})

    # Split train/validation (90/10)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Training arguments
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
        bf16=True,  # H100 mendukung bf16
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True
    )

    # Start training
    logger.info("Starting Mistral-7B LoRA fine-tuning...")
    trainer.train()
    logger.info("Training finished!")

if __name__ == "__main__":
    main()
