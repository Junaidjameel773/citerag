from types import SimpleNamespace

import pytest

from citerag.config import settings
from citerag.llm import LLM, OpenAILLM, get_llm


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
