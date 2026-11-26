from typing import Protocol

from citerag.config import settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAILLM:
    """Chat completions with the OpenAI API."""

    def __init__(self, model: str | None = None, client=None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key or None)
        self.client = client
        self.model = model or settings.llm_model

    def complete(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        return resp.choices[0].message.content or ""


def get_llm() -> LLM:
    if settings.llm_provider == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown CITERAG_LLM_PROVIDER '{settings.llm_provider}'; use openai")
