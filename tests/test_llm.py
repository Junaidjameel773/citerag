from types import SimpleNamespace

import pytest

from citerag.config import settings
from citerag.llm import GEMINI_BASE_URL, GROQ_BASE_URL, LLM, OpenAILLM, get_llm


class FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_client(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(content)))


class FakeLLM:
    def complete(self, messages):
        return messages[-1]["content"].upper()


def test_fake_provider_satisfies_protocol():
    llm: LLM = FakeLLM()
    assert llm.complete([{"role": "user", "content": "hi"}]) == "HI"


def test_openai_complete_sends_messages_and_returns_text():
    client = fake_client("Paris [1]")
    messages = [
        {"role": "system", "content": "Answer from context."},
        {"role": "user", "content": "Capital of France?"},
    ]
    assert OpenAILLM(model="m", client=client).complete(messages) == "Paris [1]"
    [call] = client.chat.completions.calls
    assert call == {"model": "m", "messages": messages, "temperature": 0}


def test_openai_empty_content_returns_empty_string():
    client = fake_client(None)
    assert OpenAILLM(client=client).complete([{"role": "user", "content": "x"}]) == ""


def test_model_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "gpt-test")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    assert OpenAILLM().model == "gpt-test"
    assert OpenAILLM(model="override", client=fake_client("")).model == "override"


def test_get_llm(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    assert isinstance(get_llm(), OpenAILLM)
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(ValueError, match="nope"):
        get_llm()


def test_default_model_without_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "")
    assert OpenAILLM(client=fake_client("")).model == "gpt-4o-mini"


def test_gemini_uses_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "llm_model", "")
    monkeypatch.setattr(settings, "gemini_api_key", "g-key")
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai")
    llm = get_llm()
    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gemini-2.5-flash"
    assert str(llm.client.base_url) == GEMINI_BASE_URL
    assert llm.client.api_key == "g-key"


def test_gemini_requires_its_own_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_llm()


def test_groq_uses_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "llm_model", "llama-custom")
    monkeypatch.setattr(settings, "groq_api_key", "gsk-key")
    llm = get_llm()
    assert llm.model == "llama-custom"
    assert str(llm.client.base_url).rstrip("/") == GROQ_BASE_URL
    assert llm.client.api_key == "gsk-key"


def test_groq_requires_its_own_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()


@pytest.mark.parametrize("url", ["http://gpu-box:11434", "http://gpu-box:11434/"])
def test_ollama_uses_configured_local_server(monkeypatch, url):
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "")
    monkeypatch.setattr(settings, "ollama_url", url)
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai")
    llm = get_llm()
    assert llm.model == "llama3.2"
    assert str(llm.client.base_url).rstrip("/") == "http://gpu-box:11434/v1"
    assert llm.client.api_key == "ollama"  # the OpenAI key is never sent to the server
