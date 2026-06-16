# Base image
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Environment variables
ENV CUDA_HOME=/usr/local/cuda

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
