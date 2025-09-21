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
```

2. Jalankan setup environment
```bash
bash setup_env.sh
```
atau manual
```bash
python3.11 -m venv venv3.11
source venv3.11/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training
```bash
accelerate launch train_mistral_lora.py

Dataset: data/logdata_wazuh_smart.jsonl
Output checkpoint: mistral-lora-finetuned/
Logs: logs/
```
