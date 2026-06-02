from pathlib import Path

import pytest

from my_video.core.analysis import summary as summary_module


class DummyMessage:
    def __init__(self, content: str):
        self.content = content


class DummyChoice:
    def __init__(self, content: str):
        self.message = DummyMessage(content)


class DummyResponse:
    def __init__(self, content: str):
        self.choices = [DummyChoice(content)]


def test_get_summary_parses_valid_json(monkeypatch, tmp_path: Path) -> None:
    txt_path = tmp_path / "optimized.txt"
    txt_path.write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(
        summary_module,
        "call_llm",
        lambda **kwargs: DummyResponse(
            '{"summary":"这是一个测试摘要","terms":[{"src":"OpenAI","tgt":"OpenAI"}]}'
        ),
    )

    result = summary_module.get_summary(txt_path, model="test-model")

    assert result == {
        "summary": "这是一个测试摘要",
        "terms": [{"src": "OpenAI", "tgt": "OpenAI"}],
    }


def test_get_summary_rejects_missing_fields(monkeypatch, tmp_path: Path) -> None:
    txt_path = tmp_path / "optimized.txt"
    txt_path.write_text("hello world", encoding="utf-8")

    monkeypatch.setattr(
        summary_module,
        "call_llm",
        lambda **kwargs: DummyResponse('{"summary":"只有摘要"}'),
    )

    with pytest.raises(ValueError, match="terms"):
        summary_module.get_summary(txt_path, model="test-model")
