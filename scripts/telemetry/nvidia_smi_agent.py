#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from statistics import mean
from typing import Any, Dict, List, Optional
from urllib import parse


class SampleStore:

    def __init__(self, gpu_indices: Optional[List[int]] = None, max_history_s: float = 3600.0):
        self._gpu_indices = gpu_indices
        self._max_history_s = max_history_s
        self._samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append_batch(self, batch: List[Dict[str, Any]]) -> None:
        if not batch:
            return

        with self._lock:
            self._samples.extend(batch)
            self._prune_locked(batch[-1]["timestamp_epoch_s"])

    def summarize_window(self, start_epoch_s: float, end_epoch_s: float) -> Dict[str, Any]:
        if end_epoch_s < start_epoch_s:
            return {
                "collector": "nvidia_smi_agent",
                "sample_count": 0,
                "node": socket.gethostname(),
                "error": "end_epoch_s must be greater than or equal to start_epoch_s",
            }

        with self._lock:
            self._prune_locked(end_epoch_s)
            window_samples = [
                sample
                for sample in self._samples
                if start_epoch_s <= sample["timestamp_epoch_s"] <= end_epoch_s
            ]

        return summarize_samples(window_samples)

    def _prune_locked(self, reference_epoch_s: float) -> None:
        if self._max_history_s <= 0:
            return

        cutoff = reference_epoch_s - self._max_history_s
        if not self._samples:
            return

        first_keep_index = 0
        for i, sample in enumerate(self._samples):
            if sample["timestamp_epoch_s"] >= cutoff:
                first_keep_index = i
                break
        else:
            self._samples.clear()
            return

        if first_keep_index > 0:
            self._samples = self._samples[first_keep_index:]


def sample_once(gpu_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return []

    timestamp_epoch_s = time.time()
    samples: List[Dict[str, Any]] = []
    for line in completed.stdout.strip().splitlines():
        if not line.strip():
            continue

        row = next(csv.reader([line]))
        if len(row) < 4:
            continue

        gpu_index = int(row[0].strip())
        if gpu_indices is not None and gpu_index not in gpu_indices:
            continue

        samples.append(
            {
                "timestamp_epoch_s": timestamp_epoch_s,
                "gpu_index": gpu_index,
                "utilization_pct": float(row[1].strip()),
                "memory_used_mb": float(row[2].strip()),
                "memory_total_mb": float(row[3].strip()),
            }
        )

    return samples


def summarize_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "collector": "nvidia_smi_agent",
            "sample_count": 0,
            "node": socket.gethostname(),
        }

    by_gpu: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_gpu[sample["gpu_index"]].append(sample)

    utilization_values = [sample["utilization_pct"] for sample in samples]
    memory_used_values = [sample["memory_used_mb"] for sample in samples]

    return {
        "collector": "nvidia_smi_agent",
        "node": socket.gethostname(),
        "sample_count": len(samples),
        "gpu_indices": sorted(by_gpu.keys()),
        "gpu_utilization_pct_avg": mean(utilization_values),
        "gpu_utilization_pct_max": max(utilization_values),
        "gpu_memory_used_mb_avg": mean(memory_used_values),
        "gpu_memory_used_mb_max": max(memory_used_values),
        "gpu_memory_used_mb_by_gpu_max": {
            gpu_index: max(sample["memory_used_mb"] for sample in gpu_samples)
            for gpu_index, gpu_samples in by_gpu.items()
        },
        "gpu_utilization_pct_by_gpu_max": {
            gpu_index: max(sample["utilization_pct"] for sample in gpu_samples)
            for gpu_index, gpu_samples in by_gpu.items()
        },
    }


class Poller:

    def __init__(self, store: SampleStore, poll_interval_s: float, gpu_indices: Optional[List[int]]):
        self._store = store
        self._poll_interval_s = poll_interval_s
        self._gpu_indices = gpu_indices
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            samples = sample_once(self._gpu_indices)
            if samples:
                self._store.append_batch(samples)
            time.sleep(self._poll_interval_s)


def make_handler(store: SampleStore):
    class Handler(BaseHTTPRequestHandler):

        def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = parse.urlparse(self.path)

            if parsed.path == "/health":
                self._send_json({"status": "ok", "node": socket.gethostname()})
                return

            if parsed.path == "/window":
                query = parse.parse_qs(parsed.query)
                try:
                    start_epoch_s = float(query["start_epoch_s"][0])
                    end_epoch_s = float(query["end_epoch_s"][0])
                except (KeyError, ValueError, IndexError):
                    self._send_json(
                        {
                            "error": "start_epoch_s and end_epoch_s query params are required",
                        },
                        status_code=400,
                    )
                    return

                payload = store.summarize_window(start_epoch_s, end_epoch_s)
                self._send_json(payload)
                return

            self._send_json({"error": "not found"}, status_code=404)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_gpu_indices(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--gpu-indices", default=None)
    parser.add_argument("--max-history-s", type=float, default=3600.0)
    args = parser.parse_args()

    gpu_indices = parse_gpu_indices(args.gpu_indices)

    store = SampleStore(gpu_indices=gpu_indices, max_history_s=args.max_history_s)
    poller = Poller(store, poll_interval_s=args.poll_interval_s, gpu_indices=gpu_indices)
    poller.start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print(f"Telemetry agent listening on {args.host}:{args.port}", flush=True)

    try:
        server.serve_forever()
    finally:
        poller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
