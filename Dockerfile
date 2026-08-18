# Base image
FROM ubuntu:22.04

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Python dependencies
# --extra-index-url is required for a CUDA-enabled torch build: vllm's own
# pin (torch==2.10.0) has no CUDA suffix, and plain PyPI only hosts a
# non-CUDA build under that exact version - the +cu128 variant only exists
# on PyTorch's own index. Confirmed empirically: the previous .sif somehow
# ended up with torch==2.9.1+cu128 despite this line never having an extra
# index in this file's history, so whatever built it relied on external,
# untracked build-environment config. Making it explicit here instead.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# Editable install of the llm_bench package itself, so `import llm_bench`
# resolves without PYTHONPATH gymnastics. Only copy what setuptools needs for
# this (pyproject.toml, README, the package dir) - the full repo (including
# data/) arrives later via the devcontainer's workspace bind mount at the
# same path, which is what actually gets used at runtime; this editable
# install just registers /app as the package location.
COPY pyproject.toml README.md ./
COPY llm_bench ./llm_bench
RUN pip3 install --no-cache-dir -e .
