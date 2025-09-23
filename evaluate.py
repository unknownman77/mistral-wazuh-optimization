import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_checkpoint(checkpoint_root="mistral-lora-finetuned"):
    """
    Cari checkpoint terbaru berdasarkan nama folder 'checkpoint-<step>'.
    """
    checkpoints = [d for d in os.listdir(checkpoint_root) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"Tidak ditemukan checkpoint di {checkpoint_root}")
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(checkpoint_root, checkpoints[-1])
    logger.info(f"Using latest checkpoint: {latest}")
    return latest

def save_output(text, idx, save_dir="eval"):
    """
    Simpan hasil evaluasi ke file .txt
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"eval_{idx+1}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved evaluation to {file_path}")

def main():
    base_model = "mistralai/Mistral-7B-v0.1"
    logger.info(f"Loading base model {base_model}...")
    
    latest_checkpoint = get_latest_checkpoint()
    
    logger.info(f"Loading tokenizer from {latest_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16  
    )
    
    logger.info(f"Applying LoRA from checkpoint {latest_checkpoint}...")
    model = PeftModel.from_pretrained(model, latest_checkpoint, torch_dtype=torch.bfloat16)
    model.eval()
    
    prompts = [
        "Generate a security alert summary for Wazuh logs:",
        "Explain what the following Wazuh log indicates:"
    ]
    
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nPrompt: {prompt}\nOutput: {text}\n{'-'*50}")
        
        save_output(text, idx, save_dir="eval")

if __name__ == "__main__":
    main()
