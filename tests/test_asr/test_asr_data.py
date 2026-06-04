import json
from pathlib import Path

import pytest

from my_video.core.asr.asr_data import ASRData, ASRDataSeg, ASRSentenceData, SentenceGroup, build_sentence_groups


class TestFromWhisperxJson:
    def test_loads_word_segments(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps(
                {
                    "word_segments": [
                        {"word": "Hello,", "start": 0.091, "end": 0.451, "score": 0.703},
                        {"word": "world", "start": 0.5, "end": 0.9, "score": 0.812},
                    ]
                }
            ),
            encoding="utf-8",
        )

        asr_data = ASRData.from_whisperx_json(str(whisperx_json))

        assert len(asr_data.segments) == 2
        assert asr_data.segments[0].text == "Hello,"
        assert asr_data.segments[0].start_time == 91
        assert asr_data.segments[0].end_time == 451
        assert asr_data.segments[1].text == "world"
        assert asr_data.segments[1].start_time == 500
        assert asr_data.segments[1].end_time == 900

    def test_empty_word_segments(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps({"word_segments": []}),
            encoding="utf-8",
        )

        asr_data = ASRData.from_whisperx_json(str(whisperx_json))

        assert asr_data.segments == []

    def test_missing_word_segments(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps({"segments": []}),
            encoding="utf-8",
        )

        asr_data = ASRData.from_whisperx_json(str(whisperx_json))

        assert asr_data.segments == []

    def test_skips_entries_with_missing_timestamps(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps(
                {
                    "word_segments": [
                        {"word": "yes", "start": 0.1, "end": 0.3},
                        {"word": "no_timestamp"},
                        {"word": "tail", "start": 0.4},
                    ]
                }
            ),
            encoding="utf-8",
        )

        asr_data = ASRData.from_whisperx_json(str(whisperx_json))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "yes"
        assert asr_data.segments[0].start_time == 100
        assert asr_data.segments[0].end_time == 300

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ASRData.from_whisperx_json("/nonexistent/path/whisperx.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text("not json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            ASRData.from_whisperx_json(str(whisperx_json))


class TestSentenceGroupGaps:
    def test_no_gap_exceeds_default_threshold(self) -> None:
        group = SentenceGroup(
            index=0,
            segments=[
                ASRDataSeg("a", 0, 100),
                ASRDataSeg("b", 150, 220),
                ASRDataSeg("c", 300, 380),
            ],
            text="a b c",
        )

        has_gap, gap_count = group.check_segment_gaps()

        assert has_gap is False
        assert gap_count == []

    def test_counts_single_gap_over_default_threshold(self) -> None:
        group = SentenceGroup(
            index=0,
            segments=[
                ASRDataSeg("a", 0, 100),
                ASRDataSeg("b", 2500, 2600),
                ASRDataSeg("c", 2700, 2800),
            ],
            text="a b c",
        )

        has_gap, gap_count = group.check_segment_gaps()

        assert has_gap is True
        assert gap_count == [1]

    def test_counts_multiple_gaps_with_custom_threshold(self) -> None:
        group = SentenceGroup(
            index=0,
            segments=[
                ASRDataSeg("a", 0, 100),
                ASRDataSeg("b", 1200, 1300),
                ASRDataSeg("c", 2700, 2800),
                ASRDataSeg("d", 5000, 5100),
            ],
            text="a b c d",
        )

        has_gap, gap_count = group.check_segment_gaps(max_gap_ms=1000)

        assert has_gap is True
        assert gap_count == [1, 2, 3]

    def test_formats_gap_markers_with_double_brackets(self) -> None:
        group = SentenceGroup(
            index=0,
            segments=[
                ASRDataSeg("hello", 0, 100),
                ASRDataSeg("world", 2500, 2600),
                ASRDataSeg("again", 2700, 2800),
            ],
            text="hello world again",
        )

        _, gap_positions = group.check_segment_gaps()

        assert group.format_with_gap_markers(gap_positions) == "hello】【world again"


class TestSentenceGroups:
    def test_build_sentence_groups_defaults_fields(self) -> None:
        segments = [
            ASRDataSeg("hello", 0, 100),
            ASRDataSeg("world.", 100, 200),
        ]

        groups = build_sentence_groups(segments)

        assert len(groups) == 1
        assert groups[0].optimized_text == ""
        assert groups[0].optimize_log == ""


class TestAsrDataJson:
    def test_to_json_and_from_json_round_trip(self, tmp_path: Path) -> None:
        asr_data = ASRData(
            [
                ASRDataSeg("hello", 0, 100),
                ASRDataSeg("world", 100, 200),
            ]
        )

        json_path = tmp_path / "asr.json"
        payload = asr_data.to_json(json_path)
        restored = ASRData.from_json(payload)

        assert json.loads(json_path.read_text(encoding="utf-8"))["segments"][0]["text"] == "hello"
        assert [seg.text for seg in restored.segments] == ["hello", "world"]

    def test_to_sentence_data_round_trip(self, tmp_path: Path) -> None:
        asr_data = ASRData(
            [
                ASRDataSeg("hello", 0, 100),
                ASRDataSeg("world.", 100, 200),
            ]
        )

        sentence_data = asr_data.to_sentence_data()
        json_path = tmp_path / "sentence.json"
        sentence_payload = sentence_data.to_json(json_path)
        restored = ASRSentenceData.from_json(sentence_payload)

        assert [group.text for group in sentence_data.sentences] == ["hello world."]
        assert [group.text for group in restored.sentences] == ["hello world."]

    def test_from_json_accepts_dict(self) -> None:
        restored = ASRSentenceData.from_json(
            {
                "sentences": [
                    {
                        "index": 0,
                        "text": "hello world",
                        "segments": [
                            {"text": "hello", "start_time": 0, "end_time": 100},
                            {"text": "world", "start_time": 100, "end_time": 200},
                        ],
                    }
                ]
            }
        )

        assert restored.sentences[0].text == "hello world"

    def test_to_txt_prefers_optimized_text_and_writes_logs(self, tmp_path: Path) -> None:
        sentence_data = ASRSentenceData(
            [
                SentenceGroup(
                    index=0,
                    segments=[ASRDataSeg("hello", 0, 100)],
                    text="hello",
                    optimized_text="HELLO",
                    optimize_log="log one",
                ),
                SentenceGroup(
                    index=1,
                    segments=[ASRDataSeg("world", 100, 200)],
                    text="world",
                ),
            ]
        )

        txt_path = tmp_path / "optimized.txt"
        content = sentence_data.to_txt(txt_path)

        assert content == "0. log one\n1. world"
        assert txt_path.read_text(encoding="utf-8") == content
