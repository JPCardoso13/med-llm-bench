# Base image
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
