from __future__ import annotations

import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ServeHandle:
    mode: str
    process: subprocess.Popen[Any]
    log_path: Path
    base_url: str
    attempted_modes: list[str]


def _first_int(value: str | None, default: int) -> int:
    if not value:
        return default
    match = re.search(r"\d+", value)
    if not match:
        return default
    return int(match.group(0))


def local_gpu_count() -> int:
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv("SINGULARITYENV_CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        indices = [i.strip() for i in cuda_visible.split(",") if i.strip()]
        if indices:
            return len(indices)

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return len(lines)
    except Exception:
        pass

    return 1


def local_gpu_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    values = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped) / 1024.0)
        except ValueError:
            continue

    if not values:
        return None

    return min(values)


def slurm_node_count() -> int:
    return _first_int(os.getenv("SLURM_NNODES") or os.getenv("SLURM_JOB_NUM_NODES"), 1)


def gpus_per_node_hint() -> int:
    return _first_int(os.getenv("SLURM_GPUS_ON_NODE") or os.getenv("SLURM_GPUS_PER_NODE"), local_gpu_count())


def cluster_gpu_capacity() -> int:
    node_count = slurm_node_count()
    gpus_per_node = gpus_per_node_hint()
    return max(1, node_count * max(1, gpus_per_node))


def _host_ip() -> str:
    env_ip = os.getenv("VLLM_HOST_IP") or os.getenv("SINGULARITYENV_VLLM_HOST_IP")
    if env_ip:
        return env_ip

    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def describe_fit(model_cfg: dict[str, Any]) -> dict[str, Any]:
    requested_tp = int(model_cfg.get("tensor_parallel_size", 1))
    local_gpus = local_gpu_count()
    cluster_gpus = cluster_gpu_capacity()
    ray_address = os.getenv("RAY_ADDRESS") or os.getenv("SINGULARITYENV_RAY_ADDRESS")

    min_vram_cfg = model_cfg.get("min_vram_gb_per_gpu")
    local_vram = local_gpu_vram_gb()

    vram_ok = True
    if min_vram_cfg is not None and local_vram is not None:
        vram_ok = float(local_vram) >= float(min_vram_cfg)

    single_possible = requested_tp <= local_gpus and vram_ok
    distributed_possible = bool(ray_address) and requested_tp <= cluster_gpus

    return {
        "requested_tp": requested_tp,
        "local_gpus": local_gpus,
        "cluster_gpus": cluster_gpus,
        "local_vram_gb": local_vram,
        "single_possible": single_possible,
        "distributed_possible": distributed_possible,
        "ray_address": ray_address,
    }


def _build_vllm_command(model_cfg: dict[str, Any], serve_port: int, mode: str) -> list[str]:
    model_id = str(model_cfg["model_id"])
    max_model_len = int(model_cfg.get("max_model_len", 4096))
    requested_tp = int(model_cfg.get("tensor_parallel_size", 1))
    gpu_mem_util = float(model_cfg.get("gpu_memory_utilization", 0.8))
    enforce_eager = bool(model_cfg.get("enforce_eager", True))

    tp_value = requested_tp
    if mode == "single_node":
        tp_value = min(requested_tp, local_gpu_count())

    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(serve_port),
        "--tensor-parallel-size",
        str(max(1, tp_value)),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
    ]

    if enforce_eager:
        cmd.append("--enforce-eager")

    if mode == "distributed":
        cmd.extend(["--distributed-executor-backend", "ray"])

    return cmd


def _wait_for_endpoint(base_url: str, process: subprocess.Popen[Any], timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    probe_url = f"{base_url}/models"

    while time.time() < deadline:
        if process.poll() is not None:
            return False

        try:
            response = urllib.request.urlopen(probe_url, timeout=2)
            if response.status == 200:
                return True
        except urllib.error.URLError:
            pass
        except Exception:
            pass

        time.sleep(2)

    return False


def stop_vllm_server(handle: ServeHandle | None) -> None:
    if handle is None:
        return

    process = handle.process
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def start_vllm_server_for_model(
    model_cfg: dict[str, Any],
    serve_port: int,
    logs_dir: str | Path,
    startup_timeout_s: int = 240,
) -> ServeHandle:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    fit = describe_fit(model_cfg)
    attempted_modes: list[str] = []

    candidate_modes: list[str] = []
    if fit["single_possible"]:
        candidate_modes.append("single_node")
    if fit["distributed_possible"]:
        candidate_modes.append("distributed")

    if not candidate_modes:
        raise RuntimeError(
            "Model cannot be scheduled with current allocation: "
            f"requested_tp={fit['requested_tp']}, local_gpus={fit['local_gpus']}, "
            f"cluster_gpus={fit['cluster_gpus']}, ray_address={fit['ray_address']}"
        )

    host_ip = _host_ip()
    base_url = f"http://{host_ip}:{serve_port}/v1"

    for mode in candidate_modes:
        attempted_modes.append(mode)
        log_path = logs_dir / f"vllm_{model_cfg.get('name', 'model')}_{mode}.log"

        cmd = _build_vllm_command(model_cfg, serve_port, mode)
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

        handle = ServeHandle(
            mode=mode,
            process=process,
            log_path=log_path,
            base_url=base_url,
            attempted_modes=attempted_modes.copy(),
        )

        if _wait_for_endpoint(base_url, process, startup_timeout_s):
            return handle

        stop_vllm_server(handle)

    raise RuntimeError(
        "Failed to start vLLM server for model "
        f"{model_cfg.get('model_id')} after attempts: {attempted_modes}"
    )
