from my_video.core.asr.asr_data import (
    ASRData,
    ASRDataSeg,
    ASRSentenceData,
    SentenceGroup,
)
from my_video.core.optimize.optimize import SubtitleOptimizer


def make_seg(text: str, start: int, end: int) -> ASRDataSeg:
    return ASRDataSeg(text=text, start_time=start, end_time=end)


class TestSubtitleOptimizerWriteBack:
    def test_equal_keeps_original_timestamps(self) -> None:
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

    def test_insert_uses_gap_between_neighbors(self) -> None:
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

    def test_optimize_log_uses_inline_diff_with_candidate_display(self) -> None:
        groups = [
            SentenceGroup(
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


class TestSubtitleOptimizerFlow:
    def test_optimize_subtitle_writes_back_to_input_and_returns_clean_object(
        self, monkeypatch
    ) -> None:
        optimizer = SubtitleOptimizer(
            thread_num=1,
            batch_num=2,
            model="test-model",
            custom_prompt="",
        )
        sentence_data = ASRSentenceData(
            [
                SentenceGroup(
                    index=0,
                    segments=[make_seg("hello", 0, 100), make_seg("world.", 160, 220)],
                    text="hello world.",
                ),
                SentenceGroup(
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

    def test_batch_sentence_groups_builds_reference_text_from_time_window(self) -> None:
        optimizer = SubtitleOptimizer(
            thread_num=1,
            batch_num=1,
            model="test-model",
            custom_prompt="",
        )
        sentence_data = ASRSentenceData(
            [
                SentenceGroup(
                    index=0,
                    segments=[make_seg("hello", 0, 100), make_seg("world.", 100, 200)],
                    text="hello world.",
                ),
                SentenceGroup(
                    index=1,
                    segments=[
                        make_seg("next", 8000, 8100),
                        make_seg("line.", 8100, 8200),
                    ],
                    text="next line.",
                ),
            ]
        )
        reference_data = ASRData(
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
        assert batches[0].start_time_ms == 0
        assert batches[0].end_time_ms == 200
        assert batches[0].reference_text == "before overlap first match second match"
        assert batches[1].reference_text == "first match second match late match"
        optimizer.stop()

    def test_batch_sentence_groups_uses_empty_reference_when_no_match(self) -> None:
        optimizer = SubtitleOptimizer(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
        )
        groups = [
            SentenceGroup(
                index=0,
                segments=[make_seg("hello", 10000, 10100)],
                text="hello",
            )
        ]
        reference_data = ASRData([make_seg("far away", 0, 1000)])

        batches = optimizer._batch_sentence_groups(groups, reference_data)

        assert batches[0].reference_text == ""
        optimizer.stop()

    def test_agent_loop_includes_plain_reference_block_without_numbering(
        self, monkeypatch
    ) -> None:
        optimizer = SubtitleOptimizer(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
        )
        captured: dict[str, object] = {}

        class DummyMessage:
            content = '{"0":"hello world."}'

        class DummyChoice:
            message = DummyMessage()

        class DummyResponse:
            choices = [DummyChoice()]

        def fake_call_llm(*, messages, model, temperature):
            captured["user_prompt"] = messages[1]["content"]
            return DummyResponse()

        monkeypatch.setattr("my_video.core.optimize.optimize.call_llm", fake_call_llm)

        result = optimizer.agent_loop(
            {"0": "hello world."}, "Reference paragraph here."
        )

        assert result == {"0": "hello world."}
        assert (
            "<reference>\nReference paragraph here.\n</reference>"
            in captured["user_prompt"]
        )
        assert "0: " not in captured["user_prompt"]
        optimizer.stop()
