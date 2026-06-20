from my_video.core.asr.asr_data import SubtitleSegment
from my_video.core.utils.helper import compact_whitespace
from my_video.core.utils.text_diff import (
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
        == "【1/4】 【2 3/∅】 5 【6/8】 【8/10】"
    )


def test_strict_diff_supports_replace_insert_delete() -> None:
    assert (
        render_inline_diff("alpha beta gamma", "alpha theta gamma extra")
        == "alpha 【beta/theta】 gamma 【∅/extra】"
    )


def test_diff_normalizes_whitespace_before_comparing() -> None:
    assert compact_whitespace("  hello   world \n ") == "hello world"
    assert render_inline_diff("hello   world", "hello world") == "hello world"


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
