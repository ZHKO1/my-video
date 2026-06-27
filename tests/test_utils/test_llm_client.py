import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
    def __init__(self, content: str, *, response_id: str = "resp-123") -> None:
        self.id = response_id
        self.choices = [_FakeChoice(content)]

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "choices": [{"message": {"content": self.choices[0].message.content}}],
        }


class _FakeMultiChoiceResponse:
    def __init__(self, contents: list[str], *, response_id: str = "resp-multi") -> None:
        self.id = response_id
        self.choices = [_FakeChoice(content) for content in contents]

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "choices": [
                {"message": {"content": choice.message.content}}
                for choice in self.choices
            ],
        }


class _FakeInvalidResponse:
    def __init__(self, *, response_id: str = "resp-invalid") -> None:
        self.id = response_id
        self.choices = []

    def model_dump(self) -> dict[str, object]:
        return {"id": self.id, "choices": []}


class _FakeNoMessageChoice:
    pass


class _FakeNoMessageResponse:
    def __init__(self, *, response_id: str = "resp-no-message") -> None:
        self.id = response_id
        self.choices = [_FakeNoMessageChoice()]


class _FakeNoneContentMessage:
    content = None


class _FakeNoneContentChoice:
    def __init__(self) -> None:
        self.message = _FakeNoneContentMessage()


class _FakeNoneContentResponse:
    def __init__(self, *, response_id: str = "resp-none-content") -> None:
        self.id = response_id
        self.choices = [_FakeNoneContentChoice()]


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


class TestCallLlm:
    def test_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            llm_client, "_call_llm_api", lambda *args, **kwargs: _FakeResponse("ok")
        )

        response = llm_client.call_llm(
            [{"role": "user", "content": "hello"}], model="demo-model"
        )

        assert response.choices[0].message.content == "ok"
        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert re.search(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} resp-123 success$",
            log_text,
            re.MULTILINE,
        )
        assert "req:\nhello\n" in log_text
        assert "res:\nchoice[0]:\nok\n" in log_text

    def test_success_last_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            llm_client,
            "_call_llm_api",
            lambda *args, **kwargs: _FakeResponse(
                "line1\\nline2", response_id="resp-last"
            ),
        )

        llm_client.call_llm(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "first line\\nsecond line"},
            ],
            model="demo-model",
        )

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert "req:\nfirst line\nsecond line\n" in log_text
        assert "system prompt" not in log_text
        assert "choice[0]:\nline1\nline2\n" in log_text

    def test_error_api(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)

        def fail(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(llm_client, "_call_llm_api", fail)

        with pytest.raises(RuntimeError, match="boom"):
            llm_client.call_llm(
                [{"role": "user", "content": "hello"}], model="demo-model"
            )

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert re.search(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - error$",
            log_text,
            re.MULTILINE,
        )
        assert "req:\nhello\n" in log_text
        assert "error:\nRuntimeError: boom" in log_text

    def test_error_invalid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            llm_client, "_call_llm_api", lambda *args, **kwargs: _FakeInvalidResponse()
        )

        with pytest.raises(ValueError, match="Invalid OpenAI API response"):
            llm_client.call_llm(
                [{"role": "user", "content": "hello"}], model="demo-model"
            )

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert re.search(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} resp-invalid error$",
            log_text,
            re.MULTILINE,
        )
        assert "req:\nhello\n" in log_text
        assert (
            "error:\nValueError: Invalid OpenAI API response: Response structure error: choices is empty."
            in log_text
        )

    def test_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        assert log_text.count(" success") == 1

    def test_append(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        responses = iter([_FakeResponse("first"), _FakeResponse("second")])
        monkeypatch.setattr(
            llm_client, "_call_llm_api", lambda *args, **kwargs: next(responses)
        )

        llm_client.call_llm([{"role": "user", "content": "one"}], model="demo-model")
        llm_client.call_llm([{"role": "user", "content": "two"}], model="demo-model")

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert log_text.count(" success") == 2
        assert "req:\none\n" in log_text
        assert "req:\ntwo\n" in log_text

    def test_choices(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            llm_client,
            "_call_llm_api",
            lambda *args, **kwargs: _FakeMultiChoiceResponse(["first", "second"]),
        )

        llm_client.call_llm([{"role": "user", "content": "hello"}], model="demo-model")

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert "choice[0]:\nfirst\n" in log_text
        assert "choice[1]:\nsecond\n" in log_text

    def test_timestamp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        work_dir = _set_work_dir(monkeypatch, tmp_path)
        monkeypatch.setattr(
            llm_client, "_call_llm_api", lambda *args, **kwargs: _FakeResponse("ok")
        )
        monkeypatch.setattr(
            llm_client,
            "format_beijing_time",
            lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=ZoneInfo("UTC"))
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .strftime("%Y-%m-%d %H:%M:%S"),
        )

        llm_client.call_llm([{"role": "user", "content": "hello"}], model="demo-model")

        log_text = (work_dir / "log" / "llm.log").read_text(encoding="utf-8")
        assert "2026-01-02 11:04:05 resp-123 success" in log_text


class TestResponseValidator:
    def test_extract_response_text(self) -> None:
        text, error = llm_client.extract_response_text(_FakeResponse("hello"))

        assert text == "hello"
        assert error == ""

    def test_extract_response_text_empty_choices(self) -> None:
        text, error = llm_client.extract_response_text(_FakeInvalidResponse())

        assert text is None
        assert error == "Response structure error: choices is empty."

    def test_extract_response_text_missing_message(self) -> None:
        text, error = llm_client.extract_response_text(_FakeNoMessageResponse())

        assert text is None
        assert error == "Response structure error: message is missing."

    def test_extract_response_text_none_content(self) -> None:
        text, error = llm_client.extract_response_text(_FakeNoneContentResponse())

        assert text is None
        assert error == "Response structure error: content is missing."

    def test_parse_json_dict(self) -> None:
        parsed, error = llm_client.parse_json_dict('{"0":"ok"}')

        assert parsed == {"0": "ok"}
        assert error == ""

    def test_parse_json_dict_not_dict(self) -> None:
        parsed, error = llm_client.parse_json_dict("[1,2]")

        assert parsed is None
        assert error == "JSON structure error: expected dict, got list."
