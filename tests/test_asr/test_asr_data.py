import json
from pathlib import Path

import pytest

from my_video.core.asr.asr_data import (
    SubtitleSegment,
    SubtitleSegments,
    SubtitleSentence,
    SubtitleSentences,
    build_sentence_list,
)


class TestFromWhisperxJson:
    def test_loads_word_segments(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps(
                {
                    "word_segments": [
                        {
                            "word": "Hello,",
                            "start": 0.091,
                            "end": 0.451,
                            "score": 0.703,
                        },
                        {"word": "world", "start": 0.5, "end": 0.9, "score": 0.812},
                    ]
                }
            ),
            encoding="utf-8",
        )

        asr_data = SubtitleSegments.from_whisperx_json(str(whisperx_json))

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

        asr_data = SubtitleSegments.from_whisperx_json(str(whisperx_json))

        assert asr_data.segments == []

    def test_missing_word_segments(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text(
            json.dumps({"segments": []}),
            encoding="utf-8",
        )

        asr_data = SubtitleSegments.from_whisperx_json(str(whisperx_json))

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

        asr_data = SubtitleSegments.from_whisperx_json(str(whisperx_json))

        assert len(asr_data.segments) == 1
        assert asr_data.segments[0].text == "yes"
        assert asr_data.segments[0].start_time == 100
        assert asr_data.segments[0].end_time == 300

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            SubtitleSegments.from_whisperx_json("/nonexistent/path/whisperx.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        whisperx_json = tmp_path / "whisperx.json"
        whisperx_json.write_text("not json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            SubtitleSegments.from_whisperx_json(str(whisperx_json))


class TestBuildSentenceList:
    def test_builds_sentences_by_sentence_end_punctuation(self) -> None:
        segments = [
            SubtitleSegment("Hello", 0, 100),
            SubtitleSegment("world,", 100, 200),
            SubtitleSegment("again.", 200, 300),
            SubtitleSegment("How", 300, 400),
            SubtitleSegment("are", 400, 500),
            SubtitleSegment("you?", 500, 600),
            SubtitleSegment("Great", 600, 700),
            SubtitleSegment("!", 700, 800),
            SubtitleSegment("你好", 800, 900),
            SubtitleSegment("世界，", 900, 1000),
            SubtitleSegment("继续", 1000, 1100),
            SubtitleSegment("测试。", 1100, 1200),
            SubtitleSegment("真的吗？", 1200, 1300),
            SubtitleSegment("当然！", 1300, 1400),
        ]

        groups = build_sentence_list(segments)

        assert [group.text for group in groups] == [
            "Hello world, again.",
            "How are you?",
            "Great !",
            "你好 世界， 继续 测试。",
            "真的吗？",
            "当然！",
        ]
        assert [group.index for group in groups] == [0, 1, 2, 3, 4, 5]
        assert all(group.optimized_text == "" for group in groups)
        assert all(group.optimize_log == "" for group in groups)


class TestSubtitleSentencesToTxt:
    def test_to_txt_prefers_optimized_text_and_writes_logs(
        self, tmp_path: Path
    ) -> None:
        sentence_data = SubtitleSentences(
            [
                SubtitleSentence(
                    index=0,
                    segments=[SubtitleSegment("hello", 0, 100)],
                    text="hello",
                    optimized_text="hello world",
                    optimize_log="hello 【∅/world】",
                ),
                SubtitleSentence(
                    index=1,
                    segments=[SubtitleSegment("world", 100, 200)],
                    text="world",
                ),
            ]
        )

        txt_path = tmp_path / "optimized.txt"
        content = sentence_data.to_txt(txt_path)

        assert content == "0. hello 【∅/world】\n1. world"
        assert txt_path.read_text(encoding="utf-8") == content


class TestSubtitleSegmentsTextIO:
    def test_to_txt_writes_joined_text(self, tmp_path: Path) -> None:
        subtitle_segments = SubtitleSegments(
            [
                SubtitleSegment("hello", 0, 100),
                SubtitleSegment("world", 100, 200),
            ]
        )

        txt_path = tmp_path / "subtitle.txt"
        content = subtitle_segments.to_txt(txt_path)

        assert content == "hello world"
        assert txt_path.read_text(encoding="utf-8") == content

    def test_segments_are_sorted_by_start_then_end_time(self) -> None:
        subtitle_segments = SubtitleSegments(
            [
                SubtitleSegment("later", 100, 300),
                SubtitleSegment("shorter", 100, 200),
                SubtitleSegment("first", 0, 50),
            ]
        )

        assert [seg.text for seg in subtitle_segments.segments] == [
            "first",
            "shorter",
            "later",
        ]

    def test_from_srt_parses_single_line_text(self) -> None:
        srt_content = """1
00:00:01,000 --> 00:00:02,500
hello world
"""

        subtitle_segments = SubtitleSegments.from_srt(srt_content)

        assert len(subtitle_segments.segments) == 1
        assert subtitle_segments.segments[0].text == "hello world"
        assert subtitle_segments.segments[0].start_time == 1000
        assert subtitle_segments.segments[0].end_time == 2500

    def test_from_srt_parses_multi_line_text_as_single_segment(self) -> None:
        srt_content = """1
00:00:03,000 --> 00:00:05,000
first line
second line
"""

        subtitle_segments = SubtitleSegments.from_srt(srt_content)

        assert len(subtitle_segments.segments) == 1
        assert subtitle_segments.segments[0].text == "first line second line"
        assert subtitle_segments.segments[0].start_time == 3000
        assert subtitle_segments.segments[0].end_time == 5000

    def test_from_srt_strips_trailing_h_markers(self) -> None:
        srt_content = """4
00:00:19,080 --> 00:00:23,280
work testing all sorts of pieces of clothing\\h
to see what they would look like "raw",\\h\\h
"""

        subtitle_segments = SubtitleSegments.from_srt(srt_content)

        assert len(subtitle_segments.segments) == 1
        assert (
            subtitle_segments.segments[0].text
            == 'work testing all sorts of pieces of clothing to see what they would look like "raw",'
        )
        assert subtitle_segments.segments[0].start_time == 19080
        assert subtitle_segments.segments[0].end_time == 23280
