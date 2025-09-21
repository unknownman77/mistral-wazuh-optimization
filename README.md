# Mistral-7B LoRA Fine-Tuning

Repository ini berisi skrip untuk fine-tuning model **Mistral-7B** menggunakan **LoRA** dengan dataset Wazuh.

## Struktur Folder

- `train_mistral_lora.py` : skrip training utama
- `data/logdata_wazuh_smart.jsonl` : dataset training
- `logs/` : folder log (akan dibuat otomatis)
- `mistral-lora-finetuned/` : checkpoint model (akan dibuat otomatis)
- `requirements.txt` : daftar library Python
- `setup_env.sh` : skrip opsional untuk setup environment

## Setup

1. Clone repo ini
```bash
git clone <repo-url>
cd mistral-wazuh-optimization
