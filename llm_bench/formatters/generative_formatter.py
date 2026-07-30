from typing import List, Optional

from llm_bench.formatters.base_formatter import BaseFormatter
from llm_bench.schemas import GenerativeSample


class GenerativeFormatter(BaseFormatter):

    def format(
        self,
        sample: GenerativeSample,
        fewshot_examples: Optional[List[GenerativeSample]] = None,
    ) -> str:
        parts = []

        if fewshot_examples and self._fewshot_template:
            for example in fewshot_examples:
                parts.append(self._render(self._fewshot_template, example))

        parts.append(self._render(self._user_turn_template, sample))

        return self._fewshot_delimiter.join(parts)
