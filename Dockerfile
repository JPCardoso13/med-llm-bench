# Base image
FROM ubuntu:22.04

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y python3-pip git libnvidia-ml1 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
