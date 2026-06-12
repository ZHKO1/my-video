import re
from pathlib import Path

import pytest

from my_video.core.llm import client as llm_client
from my_video.core.utils.cache import get_llm_cache


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]

    def model_dump(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _FakeInvalidResponse:
    def __init__(self) -> None:
        self.choices = []

    def model_dump(self) -> dict[str, object]:
        return {"choices": []}


@pytest.fixture(autouse=True)
def clear_llm_cache() -> None:
    get_llm_cache().clear()
    yield
    get_llm_cache().clear()


def _set_work_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    work_dir = tmp_path / "work"
    monkeypatch.setattr(
        llm_client,
        "load_toml_config",
        lambda: ({"work_dir": str(work_dir)}, Path("my_video.toml")),
    )
    return work_dir


def test_call_llm_writes_success_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = _set_work_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        llm_client, "_call_llm_api", lambda *args, **kwargs: _FakeResponse("ok")
    )

    response = llm_client.call_llm(
        [{"role": "user", "content": "hello"}], model="demo-model"
    )

    assert response.choices[0].message.content == "ok"
    log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
    assert re.search(r"timestamp=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", log_text)
    assert "status=success" in log_text
    assert "model=demo-model" in log_text
    assert '"content": "hello"' in log_text
    assert '"content": "ok"' in log_text


def test_call_llm_writes_error_log_when_api_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = _set_work_dir(monkeypatch, tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_client, "_call_llm_api", fail)

    with pytest.raises(RuntimeError, match="boom"):
        llm_client.call_llm([{"role": "user", "content": "hello"}], model="demo-model")

    log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
    assert "status=error" in log_text
    assert "model=demo-model" in log_text
    assert '"content": "hello"' in log_text
    assert "response=None" in log_text
    assert "error=RuntimeError: boom" in log_text


def test_call_llm_writes_error_log_when_response_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = _set_work_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(
        llm_client, "_call_llm_api", lambda *args, **kwargs: _FakeInvalidResponse()
    )

    with pytest.raises(ValueError, match="Invalid OpenAI API response"):
        llm_client.call_llm([{"role": "user", "content": "hello"}], model="demo-model")

    log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
    assert "status=error" in log_text
    assert "model=demo-model" in log_text
    assert '"content": "hello"' in log_text
    assert '"choices": []' in log_text
    assert (
        "error=ValueError: Invalid OpenAI API response: empty choices or content"
        in log_text
    )


def test_call_llm_cache_hit_does_not_write_duplicate_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = _set_work_dir(monkeypatch, tmp_path)
    calls: list[int] = []

    def fake_call(*_args, **_kwargs):
        calls.append(1)
        return _FakeResponse("ok")

    monkeypatch.setattr(llm_client, "_call_llm_api", fake_call)

    messages = [{"role": "user", "content": "hello cache"}]
    llm_client.call_llm(messages, model="demo-model")
    llm_client.call_llm(messages, model="demo-model")

    log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
    assert calls == [1]
    assert log_text.count("status=success") == 1


def test_call_llm_appends_multiple_real_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work_dir = _set_work_dir(monkeypatch, tmp_path)
    responses = iter([_FakeResponse("first"), _FakeResponse("second")])
    monkeypatch.setattr(
        llm_client, "_call_llm_api", lambda *args, **kwargs: next(responses)
    )

    llm_client.call_llm([{"role": "user", "content": "one"}], model="demo-model")
    llm_client.call_llm([{"role": "user", "content": "two"}], model="demo-model")

    log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
    assert log_text.count("status=success") == 2
    assert '"content": "one"' in log_text
    assert '"content": "two"' in log_text
