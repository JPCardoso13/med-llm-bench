from __future__ import annotations

import json
from statistics import mean
from typing import Any, Dict, List
from urllib import parse, request
from urllib.error import URLError

from llm_bench.telemetry.base_collector import TelemetryCollector


class RemoteHttpTelemetryCollector(TelemetryCollector):

    def __init__(
        self,
        endpoints: List[str],
        window_path: str = "/window",
        timeout_s: float = 1.5,
        window_padding_s: float = 0.5,
    ):
        if not endpoints:
            raise ValueError("RemoteHttpTelemetryCollector requires at least one endpoint.")

        self._endpoints = endpoints
        self._window_path = window_path
        self._timeout_s = timeout_s
        self._window_padding_s = window_padding_s

    def start_run(self, run_context: Dict[str, Any]) -> None:
        return

    def before_request(self, request_context: Dict[str, Any]) -> None:
        return

    def after_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        start_epoch_s = request_context["request_started_at_epoch_s"]
        end_epoch_s = request_context["request_finished_at_epoch_s"]

        node_results: List[Dict[str, Any]] = []
        node_errors: List[Dict[str, str]] = []

        for endpoint in self._endpoints:
            try:
                node_results.append(self._fetch_window_summary(endpoint, start_epoch_s, end_epoch_s))
            except Exception as exc:  # noqa: BLE001
                node_errors.append({"endpoint": endpoint, "error": str(exc)})

        return self._aggregate(node_results, node_errors)

    def stop_run(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "collector": "remote_http",
            "endpoint_count": len(self._endpoints),
        }

    def _fetch_window_summary(
        self,
        endpoint: str,
        start_epoch_s: float,
        end_epoch_s: float,
    ) -> Dict[str, Any]:
        base = endpoint.rstrip("/")
        padded_start_epoch_s = max(0.0, start_epoch_s - self._window_padding_s)
        padded_end_epoch_s = end_epoch_s + self._window_padding_s
        query = parse.urlencode(
            {
                "start_epoch_s": f"{padded_start_epoch_s:.6f}",
                "end_epoch_s": f"{padded_end_epoch_s:.6f}",
            }
        )
        url = f"{base}{self._window_path}?{query}"

        req = request.Request(url=url, method="GET")
        try:
            with request.urlopen(req, timeout=self._timeout_s) as response:
                payload = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"request failed for {url}: {exc}") from exc

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON from {url}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError(f"unexpected payload type from {url}: {type(parsed).__name__}")
        return parsed

    def _aggregate(
        self,
        node_results: List[Dict[str, Any]],
        node_errors: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        sample_counts = [int(result.get("sample_count", 0)) for result in node_results]

        util_max_values = [
            float(result["gpu_utilization_pct_max"])
            for result in node_results
            if "gpu_utilization_pct_max" in result
        ]
        util_avg_values = [
            float(result["gpu_utilization_pct_avg"])
            for result in node_results
            if "gpu_utilization_pct_avg" in result
        ]
        mem_max_values = [
            float(result["gpu_memory_used_mb_max"])
            for result in node_results
            if "gpu_memory_used_mb_max" in result
        ]
        mem_avg_values = [
            float(result["gpu_memory_used_mb_avg"])
            for result in node_results
            if "gpu_memory_used_mb_avg" in result
        ]

        aggregated: Dict[str, Any] = {
            "collector": "remote_http",
            "node_count_configured": len(self._endpoints),
            "node_count_reporting": len(node_results),
            "node_results": node_results,
            "sample_count_total": sum(sample_counts),
        }

        if node_errors:
            aggregated["node_errors"] = node_errors

        if util_max_values:
            aggregated["gpu_utilization_pct_max_cluster"] = max(util_max_values)
        if util_avg_values:
            aggregated["gpu_utilization_pct_avg_cluster"] = mean(util_avg_values)
        if mem_max_values:
            aggregated["gpu_memory_used_mb_max_cluster"] = max(mem_max_values)
        if mem_avg_values:
            aggregated["gpu_memory_used_mb_avg_cluster"] = mean(mem_avg_values)

        return aggregated
