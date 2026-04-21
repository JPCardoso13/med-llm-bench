from abc import ABC, abstractmethod
from typing import Any, Dict


class TelemetryCollector(ABC):

    @abstractmethod
    def start_run(self, run_context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def before_request(self, request_context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def after_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def stop_run(self, run_context: Dict[str, Any]) -> Dict[str, Any]:
        pass
