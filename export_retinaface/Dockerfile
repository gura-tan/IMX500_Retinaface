FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/src/app

# OS 依存ライブラリ
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# pip を先に更新
RUN pip install --upgrade pip setuptools wheel

# ===== 重要ポイント =====
# NumPy を先に 1.x に固定してから TensorFlow を入れる
RUN pip install \
    "numpy==1.26.4"

# TensorFlow（CPU）
RUN pip install \
    "tensorflow==2.13.1"

# RetinaFace（keras-retinaface）
RUN pip install \
    retina-face \
    opencv-python \
    pillow

WORKDIR /usr/src/app