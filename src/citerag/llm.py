from typing import Protocol

from citerag.config import settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash"}


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> str: ...


class OpenAILLM:
    """Chat completions with the OpenAI API or any OpenAI-compatible endpoint (base_url)."""

    def __init__(
        self,
        model: str | None = None,
        client=None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key or settings.openai_api_key or None, base_url=base_url)
        self.client = client
        self.model = model or settings.llm_model or DEFAULT_MODELS["openai"]

    def complete(self, messages: list[Message]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        return resp.choices[0].message.content or ""


def get_llm() -> LLM:
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS.get(provider)
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "gemini":
        # Never fall back to the OpenAI key: it would be sent to Google.
        if not settings.gemini_api_key:
            raise ValueError("CITERAG_LLM_PROVIDER=gemini needs GEMINI_API_KEY")
        return OpenAILLM(model, api_key=settings.gemini_api_key, base_url=GEMINI_BASE_URL)
    raise ValueError(f"Unknown CITERAG_LLM_PROVIDER '{provider}'; use openai or gemini")
