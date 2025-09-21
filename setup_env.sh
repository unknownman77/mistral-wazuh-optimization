#!/bin/bash

# Buat virtual environment
python3.11 -m venv venv3.11

# Aktifkan venv
source venv3.11/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Jalankan source venv3.11/bin/activate"
