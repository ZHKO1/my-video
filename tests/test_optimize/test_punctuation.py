from my_video.core.asr.asr_data import ASRData, ASRDataSeg, SentenceGroup
from my_video.core.optimize.punctuation import PunctuationBatch, PunctuationOptimizer


def make_seg(text: str, index: int) -> ASRDataSeg:
    return ASRDataSeg(text=text, start_time=index * 100, end_time=index * 100 + 90)


def make_group(index: int, word_count: int, sentence_end: str = "") -> list[ASRDataSeg]:
    segments = [make_seg(f"w{i} ", index * 1000 + i) for i in range(word_count - 1)]
    last_text = f"w{word_count - 1}{sentence_end}" if word_count else sentence_end
    segments.append(make_seg(last_text, index * 1000 + word_count))
    return segments


class TestPunctuationOptimizerFlow:
    def test_optimize_skips_short_groups_and_batches_long_groups(self, monkeypatch) -> None:
        short_group = [make_seg("Short.", 0)]
        long_group_a = make_group(1, 300, ".")
        long_group_b = make_group(2, 250, ".")
        long_group_c = make_group(3, 100, ".")
        asr_data = ASRData(short_group + long_group_a + long_group_b + long_group_c)

        optimizer = PunctuationOptimizer(
            thread_num=1,
            model="test-model",
            max_sentence_word_count=10,
        )

        captured_batches: list[PunctuationBatch] = []

        def fake_parallel_optimize(batches: list[PunctuationBatch]) -> dict[int, str]:
            captured_batches.extend(batches)
            return {
                batches[0].groups[0].index: batches[0].groups[0].text.replace("w0 ", "w0, ", 1)
            }

        monkeypatch.setattr(optimizer, "_parallel_optimize", fake_parallel_optimize)

        optimized = optimizer.optimize(asr_data)

        assert len(captured_batches) == 2
        assert [group.index for group in captured_batches[0].groups] == [1]
        assert [group.index for group in captured_batches[1].groups] == [2, 3]
        assert asr_data.segments[1].text == "w0 "
        assert optimized.segments[0].text == "Short."
        assert optimized.segments[1].text.startswith("w0,")

        optimizer.stop()

    def test_parallel_optimize_falls_back_per_failed_batch(self, monkeypatch) -> None:
        optimizer = PunctuationOptimizer(thread_num=2, model="test-model", max_sentence_word_count=10)
        batch_a = PunctuationBatch([SentenceGroup(0, [make_seg("hello ", 0)], "hello ")])
        batch_b = PunctuationBatch([SentenceGroup(1, [make_seg("world ", 1)], "world ")])

        def fake_optimize_batch(batch: PunctuationBatch) -> dict[int, str]:
            if batch.groups[0].index == 0:
                raise RuntimeError("boom")
            return {1: "world?"}

        monkeypatch.setattr(optimizer, "_optimize_batch", fake_optimize_batch)

        result = optimizer._parallel_optimize([batch_a, batch_b])

        assert result == {0: "hello ", 1: "world?"}

        optimizer.stop()


class TestPunctuationValidation:
    def test_rewrite_validation_returns_error_message(self) -> None:
        is_valid, error = PunctuationOptimizer._is_valid_punctuation_rewrite(
            "hello world how are you",
            "hello there, how are you?",
        )

        assert is_valid is False
        assert "token 2 mismatch" in error

    def test_validate_allows_partial_keys(self) -> None:
        optimizer = PunctuationOptimizer(thread_num=1, model="test-model", max_sentence_word_count=10)

        is_valid, error = optimizer._validate_optimization_result(
            {"2": "hello world how are you", "11": "fine thanks"},
            {"2": "hello world, how are you?"},
        )

        assert is_valid is True
        assert error == ""
        optimizer.stop()

    def test_validate_rejects_extra_keys(self) -> None:
        optimizer = PunctuationOptimizer(thread_num=1, model="test-model", max_sentence_word_count=10)

        is_valid, error = optimizer._validate_optimization_result(
            {"2": "hello world how are you"},
            {"2": "hello world, how are you?", "99": "extra"},
        )

        assert is_valid is False
        assert "Extra keys" in error
        optimizer.stop()

    def test_validate_rejects_word_changes(self) -> None:
        optimizer = PunctuationOptimizer(thread_num=1, model="test-model", max_sentence_word_count=10)

        is_valid, error = optimizer._validate_optimization_result(
            {"2": "hello world how are you"},
            {"2": "hello there, how are you?"},
        )

        assert is_valid is False
        assert "disallowed edits" in error
        assert "token 2 mismatch" in error
        optimizer.stop()

    def test_validate_allows_case_changes_and_multiple_spaces(self) -> None:
        optimizer = PunctuationOptimizer(thread_num=1, model="test-model", max_sentence_word_count=10)

        is_valid, error = optimizer._validate_optimization_result(
            {"2": "hello world how are you"},
            {"2": "HELLO,   world.   HOW are you?"},
        )

        assert is_valid is True
        assert error == ""
        optimizer.stop()

    def test_validate_rejects_mid_token_punctuation_changes(self) -> None:
        optimizer = PunctuationOptimizer(thread_num=1, model="test-model", max_sentence_word_count=10)

        is_valid, error = optimizer._validate_optimization_result(
            {"2": "hello world how are you"},
            {"2": "hel,lo world how are you"},
        )

        assert is_valid is False
        assert "disallowed edits" in error
        assert "token 1 mismatch" in error
        optimizer.stop()


class TestWordLevelWriteBack:
    def test_apply_text_to_segments_attaches_punctuation_and_caps(self) -> None:
        segments = [
            make_seg("hello ", 0),
            make_seg("world ", 1),
            make_seg("how ", 2),
            make_seg("are ", 3),
            make_seg("you", 4),
        ]
        original_times = [(seg.start_time, seg.end_time) for seg in segments]

        PunctuationOptimizer._apply_text_to_segments(
            segments,
            "hello, world. How are you?",
        )

        assert [seg.text for seg in segments] == [
            "hello, ",
            "world. ",
            "How ",
            "are ",
            "you?",
        ]
        assert [(seg.start_time, seg.end_time) for seg in segments] == original_times

    def test_apply_text_to_segments_handles_multiple_spaces_between_tokens(self) -> None:
        segments = [
            make_seg("hello ", 0),
            make_seg("world ", 1),
            make_seg("how ", 2),
            make_seg("are ", 3),
            make_seg("you", 4),
        ]

        PunctuationOptimizer._apply_text_to_segments(
            segments,
            "HELLO,   world.   HOW are you?",
        )

        assert [seg.text for seg in segments] == [
            "HELLO, ",
            "world. ",
            "HOW ",
            "are ",
            "you?",
        ]
