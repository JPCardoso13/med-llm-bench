from typing import Any, Dict, List, Optional
from jinja2 import BaseLoader, Environment
from llm_bench.formatters.base_formatter import BaseFormatter
from llm_bench.schemas import MCQSample


class MCQFormatter(BaseFormatter):

    def __init__(
        self,
        system_prompt: str,
        user_turn_template: str,
        fewshot_template: Optional[str] = None,
        fewshot_delimiter: str = "\n\n",
        fewshot_header: Optional[str] = None,
    ):
        self._system_prompt = system_prompt
        self._user_turn_template = user_turn_template
        self._fewshot_template = fewshot_template
        self._fewshot_delimiter = fewshot_delimiter
        self._fewshot_header = fewshot_header
        self._env = Environment(loader=BaseLoader())

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def format(
        self,
        sample: MCQSample,
        fewshot_examples: Optional[List[MCQSample]] = None,
    ) -> str:
        parts = []

        if fewshot_examples and self._fewshot_template:
            fewshot_blocks = [
                self._render(self._fewshot_template, ex)
                for ex in fewshot_examples
            ]
            fewshot_str = self._fewshot_delimiter.join(fewshot_blocks)
            if self._fewshot_header:
                fewshot_str = self._fewshot_header + fewshot_str
            parts.append(fewshot_str)

        parts.append(self._render(self._user_turn_template, sample))
        return self._fewshot_delimiter.join(parts)

    def _render(self, template_str: str, sample: MCQSample) -> str:
        template = self._env.from_string(template_str)
        return template.render(**sample.model_dump())