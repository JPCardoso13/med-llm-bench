from typing import List, Optional
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
        super().__init__(system_prompt, user_turn_template, fewshot_template, fewshot_delimiter)
        self._fewshot_header = fewshot_header

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