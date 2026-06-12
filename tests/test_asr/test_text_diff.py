from my_video.core.asr.asr_data import ASRDataSeg
from my_video.core.asr.text_diff import (
    normalize_whitespace,
    render_inline_diff,
    rewrite_segments_with_timestamps,
)


def make_seg(text: str, start: int, end: int) -> ASRDataSeg:
    return ASRDataSeg(text=text, start_time=start, end_time=end)


def test_strict_diff_detects_case_changes() -> None:
    assert render_inline_diff(
        "Hello world", "hello world", mode="strict", display="reference"
    ) == ("【Hello/hello】 world")


def test_strict_diff_detects_punctuation_changes() -> None:
    assert (
        render_inline_diff("USB", "USB.", mode="strict", display="reference")
        == "【USB/USB.】"
    )


def test_strict_diff_supports_replace_insert_delete() -> None:
    assert (
        render_inline_diff(
            "alpha beta gamma",
            "alpha theta gamma extra",
            mode="strict",
            display="reference",
        )
        == "alpha 【beta/theta】 gamma 【∅/extra】"
    )


def test_diff_normalizes_whitespace_before_comparing() -> None:
    assert normalize_whitespace("  hello   world \n ") == "hello world"
    assert (
        render_inline_diff(
            "hello   world",
            "hello world",
            mode="strict",
            display="reference",
        )
        == "hello world"
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
