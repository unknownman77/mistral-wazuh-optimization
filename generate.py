import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HUGGINGFACE_TOKEN = os.environ.get("TOKEN_HERE")

def get_latest_checkpoint(checkpoint_root="mistral-lora-finetuned"):
    """
    Finds the latest checkpoint folder based on the checkpoint number.
    """
    checkpoints = [d for d in os.listdir(checkpoint_root) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_root}")
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(checkpoint_root, checkpoints[-1])
    logger.info(f"Using latest checkpoint: {latest}")
    return latest

# Load the base model and tokenizer
base_model_path = "mistralai/Mistral-7B-v0.1"
latest_checkpoint_path = get_latest_checkpoint()

logger.info(f"Loading tokenizer from {latest_checkpoint_path}...")
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Loading base model from {base_model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    token=TOKEN_HERE
)

# Load the LoRA adapters
logger.info(f"Applying LoRA adapters from checkpoint {latest_checkpoint_path}...")
model = PeftModel.from_pretrained(model, latest_checkpoint_path)
model.eval()

def make_prompt(log_text):
    return f"""### Instruction:
Generate a Wazuh rule XML that detects the suspicious events in the logs below, include group, description, and a reasonable id.

### Logs:
{log_text}

### Response:
"""

def generate_rule(log_text, max_new_tokens=512):
    prompt = make_prompt(log_text)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
    out = tokenizer.decode(gen[0], skip_special_tokens=True)
    if '### Response:' in out:
        return out.split('### Response:')[-1].strip()
    return out
