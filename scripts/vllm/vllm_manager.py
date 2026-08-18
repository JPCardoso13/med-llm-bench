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
    max_model_len = int(model_cfg.get("max_model_len", 8192))
    gpu_mem_util = float(model_cfg.get("gpu_memory_utilization", 0.8))
    # vLLM's own default (4GiB per TP rank) totals num_gpus x 4GiB against a
    # single node's CPU RAM - on 16GB-GPU/31GB-RAM nodes that overshoots at
    # TP>=8 (32GiB requested vs ~31GiB available). 2GiB/rank keeps the total
    # comfortably under node RAM even at TP=8, while still giving KV-cache
    # overflow a real buffer. Override per-model via swap_space_gb if needed.
    swap_space_gb = float(model_cfg.get("swap_space_gb", 2))
    # vLLM's default max_num_seqs=256 sizes its dummy-request sampler warmup
    # pass accordingly, and that warmup's peak memory can exceed the
    # steady-state weight+KV-cache budget even when gpu_memory_utilization
    # looks fine on paper (observed: 7.11GiB weights + 5.03GiB KV cache should
    # fit, but the 256-wide warmup batch OOM'd anyway). This project's
    # SequentialRunner has no concurrency at all (one request at a time), so
    # there's no real workload that needs a large value here.
    max_num_seqs = int(model_cfg.get("max_num_seqs", 8))

    launcher = str(Path(__file__).parent / "vllm_launcher.py")
    cmd = [
        "python3",
        launcher,
        "--model",
        model_id,
        "--served-model-name",
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
        "--swap-space",
        str(swap_space_gb),
        "--max-num-seqs",
        str(max_num_seqs),
    ]

    chat_template = model_cfg.get("chat_template")
    if chat_template:
        # vLLM's --chat-template accepts either a file path or a literal
        # template string. Repo-relative paths (matching every other config
        # reference in this codebase, e.g. orchestrator.py's TASKS_DIR)
        # resolve fine as-is since the process cwd is always the repo root.
        cmd.extend(["--chat-template", chat_template])

    if model_cfg.get("enforce_eager", True):
        cmd.append("--enforce-eager")

    if model_cfg.get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

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


def start_vllm(model_cfg: Dict[str, Any], model_name: str, port: int = 8000, logs_dir: str | Path = "logs/vllm", timeout_s: int = 240) -> VLLMHandle:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    ray_addr = os.getenv("RAY_ADDRESS") or os.getenv("SINGULARITYENV_RAY_ADDRESS")
    node_count = int(os.getenv("LLM_NODE_COUNT") or os.getenv("SINGULARITYENV_LLM_NODE_COUNT") or "1")
    tp = int(model_cfg.get("tensor_parallel_size", 1))
    distributed = bool(ray_addr) and node_count > 1 and tp > 1
    mode = "distributed" if distributed else ("multi_gpu" if tp > 1 else "single")

    cmd = _build_cmd(model_cfg, port, distributed)
    log_path = logs_dir / f"vllm_{model_name}_{mode}.log"

    env = os.environ.copy()

    with open(log_path, "w", encoding="utf-8") as lf:
        process = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
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
