from my_video.core.asr.asr_data import (
    SubtitleSegment,
    SubtitleSegments,
    SubtitleSentence,
    SubtitleSentences,
)
from my_video.core.optimize.optimize import MAX_STEPS, SubtitleOptimizer
from tests.helpers.agent_loop import (
    install_warnings,
    make_response,
    make_response_without_message,
    sequence_call_llm,
)


def make_seg(text: str, start: int, end: int) -> SubtitleSegment:
    return SubtitleSegment(text=text, start_time=start, end_time=end)


class TestWriteBack:
    def test_equal(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 100, 220),
        ]

        rewritten = SubtitleOptimizer._rewrite_group_segments(segments, "HELLO world")

        assert [seg.text for seg in rewritten] == ["HELLO", "world"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 220),
        ]

    def test_insert_gap(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 160, 260),
        ]

        rewritten = SubtitleOptimizer._rewrite_group_segments(
            segments, "hello brave new world"
        )

        assert [seg.text for seg in rewritten] == ["hello", "brave", "new", "world"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 130),
            (130, 160),
            (160, 260),
        ]

    def test_insert_start(self) -> None:
        segments = [
            make_seg("identify", 100, 180),
            make_seg("more", 180, 250),
        ]

        rewritten = SubtitleOptimizer._rewrite_group_segments(
            segments, "It'll identify more"
        )

        assert [seg.text for seg in rewritten] == ["It'll", "identify", "more"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (100, 100),
            (100, 180),
            (180, 250),
        ]

    def test_insert_end(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 100, 220),
        ]

        rewritten = SubtitleOptimizer._rewrite_group_segments(
            segments, "hello world again"
        )

        assert [seg.text for seg in rewritten] == ["hello", "world", "again"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 220),
            (220, 220),
        ]

    def test_log(self) -> None:
        groups = [
            SubtitleSentence(
                index=0,
                segments=[
                    make_seg("alpha", 0, 100),
                    make_seg("beta", 100, 200),
                    make_seg("gamma", 200, 300),
                    make_seg("delta", 300, 400),
                    make_seg("epsilon", 400, 500),
                    make_seg("zeta", 500, 600),
                    make_seg("eta", 600, 700),
                ],
                text="alpha beta gamma delta epsilon zeta eta",
            )
        ]

        rewritten = SubtitleOptimizer._write_back_groups(
            groups,
            {"0": "alpha beta theta delta epsilon"},
        )

        assert groups[0].optimized_text == "alpha beta theta delta epsilon"
        assert (
            groups[0].optimize_log
            == "alpha beta 【gamma/theta】 delta epsilon 【zeta eta/∅】"
        )
        assert rewritten.sentences[0].text == "alpha beta theta delta epsilon"
        assert rewritten.sentences[0].optimized_text == ""
        assert rewritten.sentences[0].optimize_log == ""


class TestValidate:
    def test_keys_missing(self) -> None:
        optimizer = SubtitleOptimizer(1, 2, "test-model", "")

        is_valid, error = optimizer._validate_optimization_result(
            {"0": "hello", "1": "world"},
            {"0": "hello"},
        )

        assert not is_valid
        assert "Missing keys: ['1']" in error
        optimizer.stop()

    def test_keys_extra(self) -> None:
        optimizer = SubtitleOptimizer(1, 2, "test-model", "")

        is_valid, error = optimizer._validate_optimization_result(
            {"0": "hello"},
            {"0": "hello", "1": "world"},
        )

        assert not is_valid
        assert "Extra keys: ['1']" in error
        optimizer.stop()

    def test_similarity_short(self) -> None:
        optimizer = SubtitleOptimizer(1, 2, "test-model", "")

        is_valid, error = optimizer._validate_optimization_result(
            {"0": "alpha beta gamma"},
            {"0": "alpha theta gamma extra"},
        )

        assert is_valid
        assert error == ""
        optimizer.stop()

    def test_similarity_long(self) -> None:
        optimizer = SubtitleOptimizer(1, 2, "test-model", "")

        is_valid, error = optimizer._validate_optimization_result(
            {
                "0": "this is a longer sentence that should keep most of the original wording for subtitle optimization validation"
            },
            {"0": "completely rewritten content with a very different meaning and structure"},
        )

        assert not is_valid
        assert "Your optimizations changed the text too much" in error
        optimizer.stop()


class TestAgentLoop:
    def test_reference(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        captured: dict[str, object] = {}

        def fake_call_llm(*, messages, model, temperature):
            captured["user_prompt"] = messages[1]["content"]
            return make_response('{"0":"hello world."}')

        monkeypatch.setattr("my_video.core.optimize.optimize.call_llm", fake_call_llm)

        result = optimizer.agent_loop(
            {"0": "hello world."}, "Reference paragraph here."
        )

        assert result == {"0": "hello world."}
        assert (
            "<reference>\nReference paragraph here.\n</reference>"
            in captured["user_prompt"]
        )
        optimizer.stop()

    def test_empty(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response("   ", response_id="opt-empty"),
                    make_response('{"0":"hello world"}', response_id="opt-ok"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello world"}, "")

        assert result == {"0": "hello world"}
        assert "优化验证失败[opt-empty]" in warnings[0]
        assert "Response content is empty." in warnings[0]
        optimizer.stop()

    def test_not_dict(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response("[1, 2]", response_id="opt-list"),
                    make_response('{"0":"hello world"}', response_id="opt-ok"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello world"}, "")

        assert result == {"0": "hello world"}
        assert "JSON structure error: expected dict, got list." in warnings[0]
        optimizer.stop()

    def test_keys_missing(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"hello"}', response_id="opt-missing"),
                    make_response('{"0":"hello","1":"world"}', response_id="opt-ok"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello", "1": "world"}, "")

        assert result == {"0": "hello", "1": "world"}
        assert "Missing keys: ['1']" in warnings[0]
        optimizer.stop()

    def test_keys_extra(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"hello","1":"extra"}', response_id="opt-extra"),
                    make_response('{"0":"hello"}', response_id="opt-ok"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello"}, "")

        assert result == {"0": "hello"}
        assert "Extra keys: ['1']" in warnings[0]
        optimizer.stop()

    def test_warn(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"changed too much"}', response_id="opt-resp-1"),
                    make_response('{"0":"hello world"}', response_id="opt-resp-2"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello world"}, "")

        assert result == {"0": "hello world"}
        assert (
            "优化验证失败[opt-resp-1]，开始反馈循环 (第1次尝试): Key '0': similarity"
            in warnings[0]
        )
        optimizer.stop()

    def test_max_steps(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response('{"0":"changed too much"}', response_id=f"opt-{i}")
                    for i in range(MAX_STEPS)
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello world"}, "")

        assert result == {"0": "changed too much"}
        assert warnings[-1] == f"Max attempts reached({MAX_STEPS})，returning last result"
        optimizer.stop()

    def test_response_shape(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        warnings = install_warnings(
            monkeypatch, "my_video.core.optimize.optimize.output.warn"
        )
        monkeypatch.setattr(
            "my_video.core.optimize.optimize.call_llm",
            sequence_call_llm(
                [
                    make_response(response_id="opt-empty-choices", choices=[]),
                    make_response_without_message(response_id="opt-no-message"),
                    make_response(None, response_id="opt-no-content"),
                ]
            ),
        )

        result = optimizer.agent_loop({"0": "hello world"}, "")

        assert result == {"0": "hello world"}
        assert "choices is empty" in warnings[0]
        assert "message is missing" in warnings[1]
        assert "content is missing" in warnings[2]
        assert warnings[-1] == f"Max attempts reached({MAX_STEPS})，returning last result"
        optimizer.stop()


class TestFlow:
    def test_write_back(self, monkeypatch) -> None:
        optimizer = SubtitleOptimizer(1, 2, "test-model", "")
        sentence_data = SubtitleSentences(
            [
                SubtitleSentence(
                    index=0,
                    segments=[make_seg("hello", 0, 100), make_seg("world.", 160, 220)],
                    text="hello world.",
                ),
                SubtitleSentence(
                    index=1,
                    segments=[make_seg("good", 220, 300), make_seg("day", 300, 420)],
                    text="good day",
                ),
            ]
        )

        monkeypatch.setattr(
            optimizer,
            "_parallel_optimize",
            lambda batches: {"0": "hello brave world.", "1": "good day"},
        )

        optimized = optimizer.optimize_subtitle(sentence_data)

        assert sentence_data.sentences[0].optimized_text == "hello brave world."
        assert sentence_data.sentences[0].optimize_log == "hello 【∅/brave】 world."
        assert sentence_data.sentences[1].optimized_text == ""
        assert optimized is not sentence_data
        assert [group.text for group in optimized.sentences] == [
            "hello brave world.",
            "good day",
        ]
        assert optimized.sentences[0].optimized_text == ""
        assert optimized.sentences[0].optimize_log == ""
        optimizer.stop()

    def test_batch_reference(self) -> None:
        optimizer = SubtitleOptimizer(1, 1, "test-model", "")
        sentence_data = SubtitleSentences(
            [
                SubtitleSentence(
                    index=0,
                    segments=[make_seg("hello", 0, 100), make_seg("world.", 100, 200)],
                    text="hello world.",
                ),
                SubtitleSentence(
                    index=1,
                    segments=[
                        make_seg("next", 8000, 8100),
                        make_seg("line.", 8100, 8200),
                    ],
                    text="next line.",
                ),
            ]
        )
        reference_data = SubtitleSegments(
            [
                make_seg("before overlap", 0, 50),
                make_seg("first match", 3000, 4000),
                make_seg("second match", 5000, 7000),
                make_seg("late match", 12500, 13000),
                make_seg("too late", 14000, 15000),
            ]
        )

        batches = optimizer._batch_sentence_groups(
            sentence_data.sentences, reference_data
        )

        assert len(batches) == 2
        assert batches[0].reference_text == "before overlap first match second match"
        assert batches[1].reference_text == "first match second match late match"
        optimizer.stop()

    def test_batch_reference_empty(self) -> None:
        optimizer = SubtitleOptimizer(1, 20, "test-model", "")
        groups = [
            SubtitleSentence(
                index=0,
                segments=[make_seg("hello", 10000, 10100)],
                text="hello",
            )
        ]
        reference_data = SubtitleSegments([make_seg("far away", 0, 1000)])

        batches = optimizer._batch_sentence_groups(groups, reference_data)

        assert batches[0].reference_text == ""
        optimizer.stop()
