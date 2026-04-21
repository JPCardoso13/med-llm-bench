from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

from llm_bench.telemetry.base_collector import TelemetryCollector


class NvidiaSmiTelemetryCollector(TelemetryCollector):

    def __init__(self, gpu_indices: Optional[List[int]] = None, poll_interval_s: float = 0.5):
        self._gpu_indices = gpu_indices
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._samples: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_run(self, run_context: Dict[str, Any]) -> None:
        if shutil.which("nvidia-smi") is None:
            raise RuntimeError("nvidia-smi was not found, but NvidiaSmiTelemetryCollector was selected.")

        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def before_request(self, request_context: Dict[str, Any]) -> None:
        return

    def after_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = request_context["request_started_at_perf_counter"]
        end_time = request_context["request_finished_at_perf_counter"]

        with self._lock:
            window_samples = [
                sample
                for sample in self._samples
                if start_time <= sample["timestamp_perf_counter"] <= end_time
            ]

        return self._summarize(window_samples)

    def stop_run(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            return self._summarize(self._samples)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._append_sample_batch()
            time.sleep(self._poll_interval_s)

    def _append_sample_batch(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            return

        timestamp_perf_counter = time.perf_counter()
        lines = completed.stdout.strip().splitlines()

        batch: List[Dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue

            row = next(csv.reader([line]))
            if len(row) < 4:
                continue

            gpu_index = int(row[0].strip())
            if self._gpu_indices is not None and gpu_index not in self._gpu_indices:
                continue

            batch.append(
                {
                    "timestamp_perf_counter": timestamp_perf_counter,
                    "gpu_index": gpu_index,
                    "utilization_pct": float(row[1].strip()),
                    "memory_used_mb": float(row[2].strip()),
                    "memory_total_mb": float(row[3].strip()),
                }
            )

        if not batch:
            return

        with self._lock:
            self._samples.extend(batch)

    def _summarize(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            return {
                "collector": "nvidia_smi",
                "sample_count": 0,
            }

        by_gpu: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            by_gpu[sample["gpu_index"]].append(sample)

        utilization_values = [sample["utilization_pct"] for sample in samples]
        memory_used_values = [sample["memory_used_mb"] for sample in samples]

        return {
            "collector": "nvidia_smi",
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
