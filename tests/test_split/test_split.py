from my_video.core.asr.asr_data import SubtitleSegment, SubtitleSentence
from my_video.core.split.split import MAX_STEPS, SubtitleSplitter
from tests.helpers.agent_loop import (
    install_warnings,
    make_response,
    make_response_without_message,
    sequence_call_llm,
)


class TestBuildLines:
    def test_indexes(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        sentence = SubtitleSentence(
            index=3,
            segments=[
                SubtitleSegment("hello", 0, 100),
                SubtitleSegment("brave", 100, 200),
                SubtitleSegment("world", 200, 300),
            ],
            text="hello brave world",
        )

        lines = splitter._build_subtitle_lines(
            sentence,
            ["hello brave", "world"],
            next_line_index=7,
        )

        assert [line.sentence_index for line in lines] == [3, 3]
        assert [line.sentence_splited_line_index for line in lines] == [0, 1]
        assert [line.line_index for line in lines] == [7, 8]
        splitter.stop()


class TestSimilarity:
    def test_long_text(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        similarity = splitter._text_similarity(
            "this is a very long english sentence with many tokens and only two removed words for testing similarity behavior",
            "this is a very long english sentence with many tokens and removed words for testing similarity behavior",
        )

        assert similarity > 0.9
        splitter.stop()

    def test_case_punctuation(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        similarity = splitter._text_similarity(
            "Hello, WORLD!",
            "hello world",
        )

        assert similarity == 1.0
        splitter.stop()


class TestValidate:
    def test_not_dict(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        is_valid, error = splitter._validate_json([], {"0": "hello world"})

        assert not is_valid
        assert "expected dict" in error
        splitter.stop()

    def test_keys_missing(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        is_valid, error = splitter._validate_json({"1": "hello"}, {"0": "hello"})

        assert not is_valid
        assert "response keys do not match request keys" in error
        splitter.stop()

    def test_value_not_string(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        is_valid, error = splitter._validate_json({"0": 1}, {"0": "hello"})

        assert not is_valid
        assert "must be a string" in error
        splitter.stop()

    def test_empty_parts(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        is_valid, error = splitter._validate_json({"0": "<br>   <br>"}, {"0": "hello"})

        assert not is_valid
        assert "at least one non-empty part" in error
        splitter.stop()

    def test_content_changed(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )

        is_valid, error = splitter._validate_split_result(
            {"0": "hello world"},
            {"0": ["completely different text"]},
        )

        assert not is_valid
        assert "Content changed too much" in error
        splitter.stop()

    def test_length_limit(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=2,
        )

        is_valid, error = splitter._validate_split_result(
            {"0": "hello brave world"},
            {"0": ["hello brave world"]},
        )

        assert not is_valid
        assert "Length violations for key 0" in error
        splitter.stop()


class TestAgentLoop:
    def test_empty(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        warnings = install_warnings(monkeypatch, "my_video.core.split.split.output.warn")
        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            sequence_call_llm(
                [
                    make_response(""),
                    make_response('{"0":"hello<br>world"}', response_id="split-ok"),
                ]
            ),
        )

        result = splitter._agent_loop({"0": "hello world"})

        assert result == {"0": ["hello", "world"]}
        assert "分割验证失败[resp-1]" in warnings[0]
        assert "Response content is empty." in warnings[0]
        splitter.stop()

    def test_invalid_json(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        warnings = install_warnings(monkeypatch, "my_video.core.split.split.output.warn")
        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            sequence_call_llm(
                [
                    make_response("[1, 2]", response_id="split-bad"),
                    make_response('{"0":"hello<br>world"}', response_id="split-ok"),
                ]
            ),
        )

        result = splitter._agent_loop({"0": "hello world"})

        assert result == {"0": ["hello", "world"]}
        assert "分割验证失败[split-bad]" in warnings[0]
        assert "expected dict" in warnings[0]
        splitter.stop()

    def test_warn(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        warnings = install_warnings(monkeypatch, "my_video.core.split.split.output.warn")
        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"rewritten text"}', response_id="split-resp-1"),
                    make_response('{"0":"hello<br>world"}', response_id="split-resp-2"),
                ]
            ),
        )

        result = splitter._agent_loop({"0": "hello world"})

        assert result == {"0": ["hello", "world"]}
        assert (
            "分割验证失败[split-resp-1]，开始反馈循环 (第1次尝试): Content changed too much"
            in warnings[0]
        )
        splitter.stop()

    def test_max_steps(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=1,
        )
        warnings = install_warnings(monkeypatch, "my_video.core.split.split.output.warn")
        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"hello<br>brave world"}', response_id=f"split-{i}")
                    for i in range(MAX_STEPS)
                ]
            ),
        )

        result = splitter._agent_loop({"0": "hello brave world"})

        assert result == {"0": ["hello", "brave world"]}
        assert warnings[-1] == f"Max attempts reached({MAX_STEPS})，returning last result"
        splitter.stop()

    def test_response_shape(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        warnings = install_warnings(monkeypatch, "my_video.core.split.split.output.warn")
        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            sequence_call_llm(
                [
                    make_response(response_id="split-empty-choices", choices=[]),
                    make_response_without_message(response_id="split-no-message"),
                    make_response(None, response_id="split-no-content"),
                ]
            ),
        )

        result = splitter._agent_loop({"0": "hello world"})

        assert result == {"0": ["hello world"]}
        assert "choices is empty" in warnings[0]
        assert "message is missing" in warnings[1]
        assert "content is missing" in warnings[2]
        assert warnings[-1] == f"Max attempts reached({MAX_STEPS})，returning last result"
        splitter.stop()
