# Base image
FROM ubuntu:22.04

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Editable install of the llm_bench package itself, so `import llm_bench`
# resolves without PYTHONPATH gymnastics. Only copy what setuptools needs for
# this (pyproject.toml, README, the package dir) - the full repo (including
# data/) arrives later via the devcontainer's workspace bind mount at the
# same path, which is what actually gets used at runtime; this editable
# install just registers /app as the package location.
COPY pyproject.toml README.md ./
COPY llm_bench ./llm_bench
RUN pip3 install --no-cache-dir -e .
