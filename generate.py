import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = os.environ.get('INFER_MODEL', 'mistral-lora-finetuned/checkpoint-18750')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map='auto',
    torch_dtype=torch.float16
)

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
