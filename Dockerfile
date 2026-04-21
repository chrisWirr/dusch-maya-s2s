FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY .env.example ./

RUN python3 -m pip install --upgrade pip && python3 -m pip install -e .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "maya_s2s.server:app", "--host", "0.0.0.0", "--port", "8000"]

