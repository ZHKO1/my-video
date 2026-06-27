from my_video.core.translate.llm_translator import LLMTranslator
from my_video.core.translate.types import TargetLanguage
from tests.helpers.agent_loop import (
    install_warnings,
    make_response,
    make_response_without_message,
    sequence_call_llm,
)


class TestValidate:
    def test_valid(self) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )

        is_valid, error = translator._validate_llm_response({"0": "你好"}, {"0": "hello"})

        assert is_valid
        assert error == ""
        translator.stop()

    def test_keys_missing(self) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )

        is_valid, error = translator._validate_llm_response({}, {"0": "hello"})

        assert not is_valid
        assert "Missing keys ['0']" in error
        translator.stop()

    def test_keys_extra(self) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )

        is_valid, error = translator._validate_llm_response(
            {"0": "你好", "1": "多余"},
            {"0": "hello"},
        )

        assert not is_valid
        assert "Extra keys ['1']" in error
        translator.stop()

    def test_reflect_not_dict(self) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=True,
        )

        is_valid, error = translator._validate_llm_response({"0": "你好"}, {"0": "hello"})

        assert not is_valid
        assert "value must be a dict" in error
        translator.stop()

    def test_reflect_missing_field(self) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=True,
        )

        is_valid, error = translator._validate_llm_response(
            {"0": {"thinking": "..."}} ,
            {"0": "hello"},
        )

        assert not is_valid
        assert "missing 'native_translation' field" in error
        translator.stop()


class TestAgentLoop:
    def test_invalid_json(self, monkeypatch) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )
        warnings = install_warnings(
            monkeypatch, "my_video.core.translate.llm_translator.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.call_llm",
            sequence_call_llm(
                [
                    make_response("[1, 2]", response_id="translate-bad"),
                    make_response('{"0":"你好"}', response_id="translate-ok"),
                ]
            ),
        )

        result = translator._agent_loop("system prompt", {"0": "hello"})

        assert result == {"0": "你好"}
        assert "翻译验证失败[translate-bad]" in warnings[0]
        assert "JSON structure error: expected dict, got list." in warnings[0]
        translator.stop()

    def test_warn(self, monkeypatch) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )
        warnings = install_warnings(
            monkeypatch, "my_video.core.translate.llm_translator.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.call_llm",
            sequence_call_llm(
                [
                    make_response("{}", response_id="translate-resp-1"),
                    make_response('{"0":"你好"}', response_id="translate-resp-2"),
                ]
            ),
        )

        result = translator._agent_loop("system prompt", {"0": "hello"})

        assert result == {"0": "你好"}
        assert (
            "翻译验证失败[translate-resp-1]，开始反馈循环 (第1次尝试): Missing keys ['0']"
            in warnings[0]
        )
        translator.stop()

    def test_max_steps(self, monkeypatch) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )
        warnings = install_warnings(
            monkeypatch, "my_video.core.translate.llm_translator.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.call_llm",
            sequence_call_llm(
                [make_response("{}", response_id=f"translate-{i}") for i in range(translator.MAX_STEPS)]
            ),
        )

        result = translator._agent_loop("system prompt", {"0": "hello"})

        assert result == {"0": "hello"}
        assert warnings[-1] == f"Max attempts reached({translator.MAX_STEPS})，returning last result"
        translator.stop()

    def test_response_shape(self, monkeypatch) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )
        warnings = install_warnings(
            monkeypatch, "my_video.core.translate.llm_translator.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.call_llm",
            sequence_call_llm(
                [
                    make_response(response_id="translate-empty-choices", choices=[]),
                    make_response_without_message(response_id="translate-no-message"),
                    make_response(None, response_id="translate-no-content"),
                ]
            ),
        )

        result = translator._agent_loop("system prompt", {"0": "hello"})

        assert result == {"0": "hello"}
        assert "choices is empty" in warnings[0]
        assert "message is missing" in warnings[1]
        assert "content is missing" in warnings[2]
        translator.stop()

    def test_fallback_original(self, monkeypatch) -> None:
        translator = LLMTranslator(
            thread_num=1,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model="test-model",
            custom_prompt="",
            is_reflect=False,
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.call_llm",
            sequence_call_llm(
                [make_response("[]", response_id=f"translate-{i}") for i in range(translator.MAX_STEPS)]
            ),
        )
        monkeypatch.setattr(
            "my_video.core.translate.llm_translator.output.warn",
            lambda _message: None,
        )

        result = translator._agent_loop("system prompt", {"0": "hello"})

        assert result == {"0": "hello"}
        translator.stop()
