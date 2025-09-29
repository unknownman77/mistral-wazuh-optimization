<div align="center">

# Mistral-7B LoRA Fine-Tuning

<em>Repository ini berisi skrip untuk fine-tuning model **Mistral-7B** menggunakan **LoRA** dengan dataset Wazuh.</em>

<img src="https://img.shields.io/github/license/unknownman77/mistral-wazuh-optimization?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/unknownman77/mistral-wazuh-optimization?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/unknownman77/mistral-wazuh-optimization?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/unknownman77/mistral-wazuh-optimization?style=flat&color=0080ff" alt="repo-language-count">

<em>Tech Stack:</em>

<img src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&logo=Markdown&logoColor=white" alt="Markdown">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat&logo=tqdm&logoColor=black" alt="tqdm">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<br>
<img src="https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white" alt="Docker">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white" alt="Pydantic">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">

</div>
<br>

## Setup

1. Clone repo ini
```bash
git clone https://github.com/unknownman77/mistral-wazuh-optimization.git
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

## Deploy Web App dengan Docker

### 1. Build docker image:
```bash
docker-compose up --build -d
```

### 2. Akses Web App:
```bash
http://<server-ip>:7070

[Pastikan server punya GPU NVIDIA dan nvidia-docker2.]
```

### Pastikan requirements.txt terbaru
```bash
- Include semua dependency: `transformers`, `torch`, `trl`, `peft`, `flask`, dll.  
- Gunakan versi yang stabil agar docker build tidak error.
```

### Struktur File
```bash
mistral-wazuh-optimization/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
├── .gitignore
├── train_mistral_lora.py
├── evaluate.py
├── generate.py
├── app.py
├── setup_env.sh
├── accelerate_config.yaml
├── data/
│   └── dataset
├── logs/
├── eval/
├── template/
│   └── index.html
├── static/
│   └── style.css
└── mistral-lora-finetuned/ (di .gitignore)
```

### Penjelasan File

- `train_mistral_lora.py` : skrip training utama
- `data/logdata_wazuh_smart.jsonl` : dataset training
- `logs/` : folder log (akan dibuat otomatis)
- `mistral-lora-finetuned/` : checkpoint model (akan dibuat otomatis)
- `requirements.txt` : daftar library Python
- `setup_env.sh` : skrip opsional untuk setup environment
