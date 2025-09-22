# Base image dengan CUDA & cuDNN
FROM nvidia/cuda:12.1.105-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV INFER_MODEL=/app/mistral-lora-finetuned

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git wget curl unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Setup virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip dan install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port Flask
EXPOSE 8080

# Run Flask web app
CMD ["python", "app.py"]
