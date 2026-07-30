from abc import ABC, abstractmethod
from typing import List, Optional

from jinja2 import BaseLoader, Environment
from llm_bench.schemas import GenerativeSample, MCQSample

Sample = MCQSample | GenerativeSample


class BaseFormatter(ABC):
    """Shared Jinja rendering plumbing for text-templated formatters.

    format() stays abstract and per-subclass on purpose: MCQ and generative
    assemble their fewshot/user-turn parts differently today, and a future
    task type (e.g. multi-turn, vision) may need format() to return
    something other than a single string entirely - that's a decision for
    whoever builds it, not something this base class should presume. What's
    shared here is only the templating mechanics both current subclasses
    happen to need identically.
    """

    def __init__(
        self,
        system_prompt: str,
        user_turn_template: str,
        fewshot_template: Optional[str] = None,
        fewshot_delimiter: str = "\n\n",
    ):
        self._system_prompt = system_prompt
        self._user_turn_template = user_turn_template
        self._fewshot_template = fewshot_template
        self._fewshot_delimiter = fewshot_delimiter
        self._env = Environment(loader=BaseLoader())

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def _render(self, template_str: str, sample: Sample) -> str:
        template = self._env.from_string(template_str)
        return template.render(**sample.model_dump())

    @abstractmethod
    def format(
        self,
        sample: Sample,
        fewshot_examples: Optional[List[Sample]] = None,
    ) -> str:
        """
        Format a sample into a prompt string ready to send to the model.
        Fewshot examples are optional and handled here if provided.
        """
        pass