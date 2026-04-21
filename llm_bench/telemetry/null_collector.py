from typing import Any, Dict

from llm_bench.telemetry.base_collector import TelemetryCollector


class NullTelemetryCollector(TelemetryCollector):

    def start_run(self, run_context: Dict[str, Any]) -> None:
        return

    def before_request(self, request_context: Dict[str, Any]) -> None:
        return

    def after_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def stop_run(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        return {}
