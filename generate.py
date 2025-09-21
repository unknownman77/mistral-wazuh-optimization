import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = os.environ.get('INFER_MODEL', 'outputs/mistral-lora')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', torch_dtype=torch.float16)


def make_prompt(log_text):
return f"### Instruction:\nGenerate a Wazuh rule XML that detects the suspicious events in the logs below and include group, description, and a reasonable id.\n\n### Logs:\n{log_text}\n\n### Response:\n"


def generate_rule(log_text, max_new_tokens=512):
prompt = make_prompt(log_text)
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
gen = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0)
out = tokenizer.decode(gen[0], skip_special_tokens=True)
# extract after '### Response:' if present
if '### Response:' in out:
return out.split('### Response:')[-1].strip()
return out

if __name__ == '__main__':
sample = "Oct 10 12:01 host sshd[123]: Failed password for invalid user admin from 10.0.0.5"
print(generate_rule(sample))