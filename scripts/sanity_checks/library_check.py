import torch
import sys
import subprocess

print(f"--- DIAGNOSTIC START ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# CUDA availability check
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    subprocess.run(["nvidia-smi"])
else:
    print(f"CUDA Available: No")
    sys.exit(1)

# vLLM import check
try:
    from vllm import LLM, SamplingParams
    print(f"vLLM library successfully imported.")
except ImportError as e:
    print(f"vLLM import failed: {e}")
    sys.exit(1)

print(f"--- DIAGNOSTIC END ---")
