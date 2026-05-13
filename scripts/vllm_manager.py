from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class VLLMHandle:
    process: subprocess.Popen[Any]
    base_url: str
    mode: str
    log_path: Path


def _terminate_process_tree(process: subprocess.Popen[Any], timeout_s: int = 20) -> None:
    if process.poll() is not None:
        return

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
        process.wait(timeout=timeout_s)
        return
    except Exception:
        pass

    try:
        process.terminate()
        process.wait(timeout=5)
        return
    except Exception:
        pass

    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            return

    try:
        process.wait(timeout=5)
    except Exception:
        pass


def _build_cmd(model_cfg: Dict[str, Any], port: int, distributed: bool) -> list[str]:
    model_id = str(model_cfg.get("model_id"))
    tp = int(model_cfg.get("tensor_parallel_size", 1))
    max_model_len = int(model_cfg.get("max_model_len", 4096))
    gpu_mem_util = float(model_cfg.get("gpu_memory_utilization", 0.8))

    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(max(1, tp)),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
    ]

    if model_cfg.get("enforce_eager", True):
        cmd.append("--enforce-eager")

    if distributed:
        cmd.extend(["--distributed-executor-backend", "ray"])
    elif tp > 1:
        cmd.extend(["--distributed-executor-backend", "mp"])

    return cmd


def _wait_for_ready(base_url: str, process: subprocess.Popen[Any], timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    probe = f"{base_url}/models"
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            r = urllib.request.urlopen(probe, timeout=3)
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def start_vllm(model_cfg: Dict[str, Any], port: int = 8000, logs_dir: str | Path = "logs/vllm", timeout_s: int = 240) -> VLLMHandle:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ray_addr = os.getenv("RAY_ADDRESS") or os.getenv("SINGULARITYENV_RAY_ADDRESS")
    node_count = int(os.getenv("LLM_NODE_COUNT") or os.getenv("SINGULARITYENV_LLM_NODE_COUNT") or "1")
    tp = int(model_cfg.get("tensor_parallel_size", 1))
    distributed = bool(ray_addr) and node_count > 1 and tp > 1
    mode = "distributed" if distributed else ("multi_gpu" if tp > 1 else "single")

    cmd = _build_cmd(model_cfg, port, distributed)
    log_path = logs_dir / f"vllm_{model_cfg.get('name','model')}_{mode}.log"

    with open(log_path, "w", encoding="utf-8") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )

    host = os.getenv("SINGULARITYENV_VLLM_HOST_IP") or os.getenv("VLLM_HOST_IP") or "127.0.0.1"
    base_url = f"http://{host}:{port}/v1"

    if not _wait_for_ready(base_url, process, timeout_s):
        _terminate_process_tree(process)
        raise RuntimeError(f"vLLM failed to become ready, see {log_path}")

    return VLLMHandle(process=process, base_url=base_url, mode=mode, log_path=log_path)


def stop_vllm(handle: VLLMHandle | None) -> None:
    if handle is None:
        return
    proc = handle.process
    if proc.poll() is not None:
        return
    _terminate_process_tree(proc)
