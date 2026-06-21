import json
from pathlib import Path

from my_video.core.asr.asr_data import SubtitleSegment
from my_video.core.utils.helper import compact_whitespace
from my_video.core.utils.text_diff import (
    merge_edit_opcodes,
    render_inline_diff,
    rewrite_segments_with_timestamps,
)


def make_seg(text: str, start: int, end: int) -> SubtitleSegment:
    return SubtitleSegment(text=text, start_time=start, end_time=end)


def test_strict_diff_detects_case_changes() -> None:
    assert render_inline_diff("Hello world", "hello world") == "【Hello/hello】 world"


def test_strict_diff_detects_punctuation_changes() -> None:
    assert render_inline_diff("USB", "USB.") == "【USB/USB.】"


def test_render_inline_diff() -> None:
    assert (
        render_inline_diff("1 2 3 5 6 8", "4 5 8 10")
        == "【1 2 3/4】 5 【6/∅】 8 【∅/10】"
    )


def test_strict_diff_supports_replace_insert_delete() -> None:
    assert (
        render_inline_diff("alpha beta gamma", "alpha theta gamma extra")
        == "alpha 【beta/theta】 gamma 【∅/extra】"
    )


def test_render_inline_diff_renders_merged_edit_block_as_single_marker() -> None:
    assert (
        render_inline_diff("Hit that later.", "Get to that later.")
        == "【Hit/Get to】 that later."
    )


def test_diff_normalizes_whitespace_before_comparing() -> None:
    assert compact_whitespace("  hello   world \n ") == "hello world"
    assert render_inline_diff("hello   world", "hello world") == "hello world"


def test_render_inline_diff_assets_cases() -> None:
    assets_path = Path(__file__).parent / "assets/render_inline_diff_case.json"
    cases = json.loads(assets_path.read_text(encoding="utf-8"))

    for record in cases:
        actual = render_inline_diff(record["txt"], record["optimized_text"])
        assert actual == record["result"], (
            f"render_inline_diff mismatch at index={record['index']} "
            f"txt={record['txt']!r} optimized_text={record['optimized_text']!r}"
        )


def test_rewrite_segments_with_timestamps_from_origin_text() -> None:
    segments = [
        make_seg("hello", 0, 100),
        make_seg("world", 100, 200),
    ]

    rewritten = rewrite_segments_with_timestamps(
        segments, "HELLO brave world!", mode="strict"
    )

    assert [seg.text for seg in rewritten] == ["HELLO", "brave", "world!"]
    assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
        (0, 100),
        (100, 150),
        (150, 200),
    ]


class TestMergeEditOpcodes:
    def test_keeps_all_equal_blocks(self) -> None:
        assert merge_edit_opcodes([("equal", 0, 2, 0, 2)]) == [("equal", 0, 2, 0, 2)]

    def test_keeps_single_insert(self) -> None:
        assert merge_edit_opcodes([("insert", 1, 1, 1, 3)]) == [
            ("insert", 1, 1, 1, 3)
        ]

    def test_keeps_single_delete(self) -> None:
        assert merge_edit_opcodes([("delete", 2, 4, 2, 2)]) == [
            ("delete", 2, 4, 2, 2)
        ]

    def test_keeps_single_replace(self) -> None:
        assert merge_edit_opcodes([("replace", 0, 2, 0, 1)]) == [
            ("replace", 0, 2, 0, 1)
        ]

    def test_merges_replace_delete_insert_into_replace(self) -> None:
        opcodes = [
            ("replace", 0, 1, 0, 1),
            ("delete", 1, 3, 1, 1),
            ("insert", 3, 3, 1, 2),
        ]

        assert merge_edit_opcodes(opcodes) == [("replace", 0, 3, 0, 2)]

    def test_merges_edit_block_at_start(self) -> None:
        opcodes = [
            ("insert", 0, 0, 0, 1),
            ("replace", 0, 1, 1, 2),
            ("equal", 1, 2, 2, 3),
        ]

        assert merge_edit_opcodes(opcodes) == [
            ("replace", 0, 1, 0, 2),
            ("equal", 1, 2, 2, 3),
        ]

    def test_merges_edit_block_at_end(self) -> None:
        opcodes = [
            ("equal", 0, 1, 0, 1),
            ("delete", 1, 2, 1, 1),
            ("insert", 2, 2, 1, 3),
        ]

        assert merge_edit_opcodes(opcodes) == [
            ("equal", 0, 1, 0, 1),
            ("replace", 1, 2, 1, 3),
        ]

    def test_does_not_merge_across_equal_blocks(self) -> None:
        opcodes = [
            ("delete", 0, 1, 0, 0),
            ("equal", 1, 2, 0, 1),
            ("insert", 2, 2, 1, 2),
        ]

        assert merge_edit_opcodes(opcodes) == [
            ("delete", 0, 1, 0, 0),
            ("equal", 1, 2, 0, 1),
            ("insert", 2, 2, 1, 2),
        ]

    def test_merges_consecutive_inserts(self) -> None:
        opcodes = [
            ("insert", 1, 1, 1, 2),
            ("insert", 1, 1, 2, 4),
        ]

        assert merge_edit_opcodes(opcodes) == [("insert", 1, 1, 1, 4)]

    def test_merges_consecutive_deletes(self) -> None:
        opcodes = [
            ("delete", 1, 2, 1, 1),
            ("delete", 2, 4, 1, 1),
        ]

        assert merge_edit_opcodes(opcodes) == [("delete", 1, 4, 1, 1)]
