from __future__ import annotations

from openai import OpenAI


class JudgeClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]], response_format: dict) -> str:
        completion = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=response_format,
        )
        return completion.choices[0].message.content or ""
