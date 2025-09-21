FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
# Gunakan -devel untuk build tools yang diperlukan

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install requirements first (for better caching)
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . /workspace

# Create necessary directories
RUN mkdir -p /workspace/outputs /workspace/logs /workspace/data

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Make scripts executable
RUN chmod +x /workspace/*.py

# Default command for API, override for training
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]