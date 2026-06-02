import pytest
from types import SimpleNamespace

from my_video.core.asr.asr_data import ASRDataSeg, ASRSentenceData, SentenceGroup
from my_video.core.split.split import (
    AlignWindow,
    SplitBatchContext,
    SplitPart,
    SplitRequest,
    SubtitleSplitter,
)


def make_group(
    index: int,
    words: list[str],
    *,
    translate: str = "",
) -> SentenceGroup:
    segments = [
        ASRDataSeg(text=word, start_time=i * 100, end_time=i * 100 + 50)
        for i, word in enumerate(words)
    ]
    return SentenceGroup(index=index, segments=segments, text=" ".join(words), translate_text=translate)


class TestSubtitleSplitter:
    def test_split_subtitle_keeps_short_groups_and_duplicates_translate(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        sentence_data = ASRSentenceData(
            [
                make_group(0, ["hello", "world"], translate="你好 世界"),
                make_group(1, ["one", "two", "three", "four", "five", "six"], translate="翻译"),
            ]
        )

        monkeypatch.setattr(
            splitter,
            "_request_parts_batch",
            lambda batch, payload: {
                1: [
                    SplitPart(src="one two three", translate=""),
                    SplitPart(src="four five six", translate=""),
                ]
            },
        )

        result = splitter.split_subtitle(sentence_data)

        assert [group.text for group in result.sentences] == [
            "hello world",
            "one two three",
            "four five six",
        ]
        assert [group.translate_text for group in result.sentences] == [
            "你好 世界",
            "翻译",
            "翻译",
        ]
        splitter.stop()

    def test_split_subtitle_splits_source_and_translate(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        sentence_data = ASRSentenceData(
            [
                make_group(
                    0,
                    ["one", "two", "three", "four", "five", "six"],
                    translate="uno dos tres cuatro cinco seis",
                )
            ]
        )

        monkeypatch.setattr(
            splitter,
            "_request_parts_batch",
            lambda batch, payload: {
                0: [
                    SplitPart(src="one two", translate="uno dos"),
                    SplitPart(src="three four five six", translate="tres cuatro cinco seis"),
                ]
            },
        )

        result = splitter.split_subtitle(sentence_data)

        assert [group.text for group in result.sentences] == [
            "one two",
            "three four five six",
        ]
        assert [group.translate_text for group in result.sentences] == [
            "uno dos",
            "tres cuatro cinco seis",
        ]
        assert [len(group.segments) for group in result.sentences] == [2, 4]
        splitter.stop()

    def test_split_subtitle_raises_when_only_translate_exceeds_limit(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        sentence_data = ASRSentenceData(
            [make_group(0, ["hello", "world"], translate="one two three four five six")]
        )

        with pytest.raises(ValueError, match="translate_text exceeds limit"):
            splitter.split_subtitle(sentence_data)
        splitter.stop()

    def test_splitter_parses_fenced_json(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
        )

        payload = splitter._parse_json_response(
            """```json
            [{"index": 0, "parts": [{"src": "hello", "translate": ""}]}]
            ```"""
        )

        assert payload[0]["index"] == 0
        splitter.stop()

    def test_splitter_validates_llm_response(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three"]),
                split_translate=False,
            )
        ]

        is_valid, error_message = splitter._validate_llm_response(
            batch,
            [{"index": 0, "parts": [{"src": "one two", "translate": ""}, {"src": "three", "translate": ""}]}],
        )

        assert is_valid is True
        assert error_message == ""
        splitter.stop()

    def test_validate_llm_response_accepts_whitespace_only_differences(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=8,
            max_word_count_english=8,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three", "four"]),
                split_translate=False,
            )
        ]

        is_valid, error_message = splitter._validate_llm_response(
            batch,
            [{"index": 0, "parts": [{"src": "one   two", "translate": ""}, {"src": "three four", "translate": ""}]}],
        )

        assert is_valid is True
        assert error_message == ""
        splitter.stop()

    def test_validate_llm_response_rejects_content_rewrite_with_feedback(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=8,
            max_word_count_english=8,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three", "four"]),
                split_translate=False,
            )
        ]

        is_valid, error_message = splitter._validate_llm_response(
            batch,
            [{"index": 0, "parts": [{"src": "one two", "translate": ""}, {"src": "rewritten content", "translate": ""}]}],
        )

        assert is_valid is False
        assert "Content changed too much" in error_message
        splitter.stop()

    def test_validate_llm_response_allows_empty_translate_when_not_split(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=8,
            max_word_count_english=8,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three"], translate="uno dos tres"),
                split_translate=False,
            )
        ]

        is_valid, error_message = splitter._validate_llm_response(
            batch,
            [{"index": 0, "parts": [{"src": "one two", "translate": ""}, {"src": "three", "translate": ""}]}],
        )

        assert is_valid is True
        assert error_message == ""
        splitter.stop()

    @pytest.mark.parametrize(
        ("payload", "pattern"),
        [
            ([{"parts": [{"src": "one", "translate": ""}]}], "JSON structure error"),
            ([{"index": 0, "parts": []}], "parts must be a non-empty list"),
            ([], "sentence indexes do not match"),
        ],
    )
    def test_validate_llm_response_rejects_structure_errors(self, payload, pattern) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=8,
            max_word_count_english=8,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three"]),
                split_translate=False,
            )
        ]

        is_valid, error_message = splitter._validate_llm_response(batch, payload)
        assert is_valid is False
        assert pattern in error_message
        splitter.stop()

    def test_validate_llm_structure_returns_bool_and_message(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=8,
            max_word_count_english=8,
        )
        request_map = {
            0: SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three"]),
                split_translate=False,
            )
        }

        is_valid, error_message = splitter._validate_llm_structure(
            request_map,
            [{"index": 0, "parts": [{"src": "one two", "translate": ""}, {"src": "three", "translate": ""}]}],
        )

        assert is_valid is True
        assert error_message == ""
        splitter.stop()

    def test_normalize_text_removes_spaces_for_cjk(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
        )

        assert splitter._normalize_text("你 好 世 界") == "你好世界"
        assert splitter._join_parts(["你好", "世界"]) == "你好世界"
        assert splitter._segments_to_text(
            [
                ASRDataSeg("你", 0, 10),
                ASRDataSeg("好", 10, 20),
                ASRDataSeg("世界", 20, 30),
            ]
        ) == "你好世界"
        splitter.stop()

    def test_request_parts_batch_retries_until_valid(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three"]),
                split_translate=False,
            )
        ]

        responses = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='[{"index": 0, "parts": [{"src": "bad", "translate": ""}]}]'
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='[{"index": 0, "parts": [{"src": "one two three", "translate": ""}]}]'
                            )
                        )
                    ]
                ),
            ]
        )

        calls = {"count": 0}

        def fake_call_llm(**kwargs):
            calls["count"] += 1
            return next(responses)

        monkeypatch.setattr("my_video.core.split.split.call_llm", fake_call_llm)

        result = splitter._request_parts_batch(
            batch,
            [{"index": 0, "src": "one two three", "translate": ""}],
        )

        assert calls["count"] == 2
        assert [part.src for part in result[0]] == ["one two three"]
        splitter.stop()

    def test_request_parts_batch_returns_last_structurally_valid_alignment_fallback(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three", "four", "five", "six"]),
                split_translate=False,
            )
        ]
        responses = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='[{"index": 0, "parts": [{"src": "one two three", "translate": ""}, {"src": "four five six", "translate": ""}]}]'
                            )
                        )
                    ]
                )
                for _ in range(3)
            ]
        )

        monkeypatch.setattr("my_video.core.split.split.call_llm", lambda **kwargs: next(responses))
        monkeypatch.setattr(
            splitter,
            "_validate_alignment_feasibility",
            lambda context, parts_map: (
                False,
                "Alignment issue: sentence 0 boundary is still not ideal",
                True,
            ),
        )

        result = splitter._request_parts_batch(
            batch,
            [{"index": 0, "src": "one two three four five six", "translate": ""}],
        )

        assert [part.src for part in result[0]] == ["one two three", "four five six"]
        splitter.stop()

    def test_validate_candidate_response_preserves_parts_for_alignment_fallback(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        batch = [
            SplitRequest(
                sentence_index=0,
                group=make_group(0, ["one", "two", "three", "four"]),
                split_translate=False,
            )
        ]
        context = SplitBatchContext(
            batch=batch,
            request_map={0: batch[0]},
            payload=[{"index": 0, "src": "one two three four", "translate": ""}],
        )

        monkeypatch.setattr(
            splitter,
            "_validate_alignment_feasibility",
            lambda context, parts_map: (
                False,
                "Alignment issue: sentence 0 boundary is still not ideal",
                True,
            ),
        )

        result = splitter._validate_candidate_response(
            context,
            '[{"index": 0, "parts": [{"src": "one two", "translate": ""}, {"src": "three four", "translate": ""}]}]',
        )

        assert result.is_valid is False
        assert result.alignment_fallback_only is True
        assert [part.src for part in result.parts_map[0]] == ["one two", "three four"]
        splitter.stop()

    def test_split_group_by_parts_uses_best_window_for_near_match(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        group = make_group(0, ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"])

        split_groups = splitter._split_group_by_parts(
            group,
            [
                SplitPart(src="alpha beta gamma delta!", translate=""),
                SplitPart(src="epsilon zeta", translate=""),
            ],
            duplicate_translate=True,
        )

        assert [item.text for item in split_groups] == ["alpha beta gamma delta!", "epsilon zeta"]
        assert [len(item.segments) for item in split_groups] == [4, 2]
        splitter.stop()

    def test_find_best_window_prefers_shorter_distance_on_tie(self, monkeypatch) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        group = make_group(0, ["alpha", "beta", "gamma"])

        monkeypatch.setattr(
            splitter,
            "_score_window",
            lambda **kwargs: AlignWindow(
                end_index=kwargs["end_seg_index"],
                similarity=0.95,
                distance=kwargs["end_seg_index"] - kwargs["start_seg_index"],
                length_gap=0,
            ),
        )

        window = splitter._find_best_window_for_part(
            group=group,
            start_seg_index=0,
            max_end_index=2,
            target_text="alpha",
        )

        assert window is not None
        assert window.end_index == 0
        splitter.stop()

    def test_split_group_by_parts_supports_cjk_near_match(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=4,
            max_word_count_english=4,
        )
        group = SentenceGroup(
            index=0,
            segments=[
                ASRDataSeg("今", 0, 10),
                ASRDataSeg("天", 10, 20),
                ASRDataSeg("天", 20, 30),
                ASRDataSeg("气", 30, 40),
                ASRDataSeg("很", 40, 50),
                ASRDataSeg("好", 50, 60),
            ],
            text="今天天气很好",
        )

        split_groups = splitter._split_group_by_parts(
            group,
            [SplitPart(src="今天天气！", translate=""), SplitPart(src="很好", translate="")],
            duplicate_translate=True,
        )

        assert [item.text for item in split_groups] == ["今天天气！", "很好"]
        assert [len(item.segments) for item in split_groups] == [4, 2]
        splitter.stop()

    def test_splitter_rejects_severely_distorted_alignment(self) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            model="test-model",
            max_word_count_cjk=5,
            max_word_count_english=5,
        )
        group = make_group(0, ["one", "two", "three", "four"])

        with pytest.raises(ValueError, match="split result changed source text"):
            splitter._split_group_by_parts(
                group,
                [
                    SplitPart(src="one", translate=""),
                    SplitPart(src="not matching", translate=""),
                ],
                duplicate_translate=True,
            )
        splitter.stop()
