FROM nvidia/cuda:12.1.105-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV INFER_MODEL=/app/mistral-lora-finetuned

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git wget curl unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
