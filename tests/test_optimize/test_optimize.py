from my_video.core.asr.asr_data import ASRDataSeg, ASRSentenceData, SentenceGroup
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

        rewritten = SubtitleOptimizer._rewrite_group_segments(segments, "hello brave new world")

        assert [seg.text for seg in rewritten] == ["hello", "brave", "new", "world"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 130),
            (130, 160),
            (160, 260),
        ]

    def test_optimize_logs_use_fixed_context_format(self) -> None:
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
        assert groups[0].optimize_logs == [
            "alpha beta 【 gamma / theta 】delta epsilon zeta eta",
            "alpha beta gamma delta epsilon 【 zeta eta / ∅ 】∅",
        ]
        assert rewritten.sentences[0].text == "alpha beta theta delta epsilon"
        assert rewritten.sentences[0].optimized_text == ""
        assert rewritten.sentences[0].optimize_logs == []


class TestSubtitleOptimizerFlow:
    def test_optimize_subtitle_writes_back_to_input_and_returns_clean_object(self, monkeypatch) -> None:
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
            lambda chunks: {"0": "hello brave world.", "1": "good day"},
        )

        optimized = optimizer.optimize_subtitle(sentence_data)

        assert sentence_data.sentences[0].optimized_text == "hello brave world."
        assert sentence_data.sentences[0].optimize_logs == [
            "hello 【 ∅ / brave 】world."
        ]
        assert sentence_data.sentences[1].optimized_text == ""
        assert optimized is not sentence_data
        assert [group.text for group in optimized.sentences] == [
            "hello brave world.",
            "good day",
        ]
        assert optimized.sentences[0].optimized_text == ""
        assert optimized.sentences[0].optimize_logs == []
        optimizer.stop()
