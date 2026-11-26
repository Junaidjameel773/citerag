from typing import Protocol

from citerag.config import settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
}

# Hosted providers with OpenAI-compatible APIs: base URL and the env var holding their key.
COMPATIBLE = {
    "gemini": (GEMINI_BASE_URL, "GEMINI_API_KEY"),
    "groq": (GROQ_BASE_URL, "GROQ_API_KEY"),
}


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
    if provider in COMPATIBLE:
        base_url, key_var = COMPATIBLE[provider]
        api_key = getattr(settings, key_var.lower())
        # Never fall back to the OpenAI key: it would be sent to another vendor.
        if not api_key:
            raise ValueError(f"CITERAG_LLM_PROVIDER={provider} needs {key_var}")
        return OpenAILLM(model, api_key=api_key, base_url=base_url)
    if provider == "ollama":
        # Local server with an OpenAI-compatible /v1 API; it ignores the key but the SDK needs one.
        base_url = settings.ollama_url.rstrip("/") + "/v1"
        return OpenAILLM(model, api_key="ollama", base_url=base_url)
    raise ValueError(
        f"Unknown CITERAG_LLM_PROVIDER '{provider}'; use openai, gemini, groq or ollama"
    )
