import json
from pathlib import Path

from my_video.core.asr.asr_data import SubtitleSegment
from my_video.core.utils.helper import compact_whitespace
from my_video.core.utils.text_diff import (
    merge_edit_opcodes,
    render_inline_diff,
    rewrite_sentence_segments_with_timestamps,
)


def make_seg(text: str, start: int, end: int) -> SubtitleSegment:
    return SubtitleSegment(text=text, start_time=start, end_time=end)


class TestRenderDiff:
    def test_case(self) -> None:
        assert render_inline_diff("Hello world", "hello world") == "【Hello/hello】 world"

    def test_punctuation(self) -> None:
        assert render_inline_diff("USB", "USB.") == "【USB/USB.】"

    def test_basic(self) -> None:
        assert (
            render_inline_diff("1 2 3 5 6 8", "4 5 8 10")
            == "【1 2 3/4】 5 【6/∅】 8 【∅/10】"
        )

    def test_edit_ops(self) -> None:
        assert (
            render_inline_diff("alpha beta gamma", "alpha theta gamma extra")
            == "alpha 【beta/theta】 gamma 【∅/extra】"
        )

    def test_merged_block(self) -> None:
        assert (
            render_inline_diff("Hit that later.", "Get to that later.")
            == "【Hit/Get to】 that later."
        )

    def test_whitespace(self) -> None:
        assert compact_whitespace("  hello   world \n ") == "hello world"
        assert render_inline_diff("hello   world", "hello world") == "hello world"

    def test_assets(self) -> None:
        assets_path = Path(__file__).parent / "assets/render_inline_diff_case.json"
        cases = json.loads(assets_path.read_text(encoding="utf-8"))

        for record in cases:
            actual = render_inline_diff(record["txt"], record["optimized_text"])
            assert actual == record["result"], (
                f"render_inline_diff mismatch at index={record['index']} "
                f"txt={record['txt']!r} optimized_text={record['optimized_text']!r}"
            )


class TestRewriteSegments:
    def test_equal(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 100, 200),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(segments, "HELLO world")

        assert [seg.text for seg in rewritten] == ["HELLO", "world"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 200),
        ]

    def test_insert_start(self) -> None:
        segments = [
            make_seg("identify", 100, 180),
            make_seg("more", 180, 250),
            make_seg("of", 250, 290),
            make_seg("the", 290, 330),
            make_seg("code.", 330, 420),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(
            segments, "It'll identify more of the code."
        )

        assert [seg.text for seg in rewritten] == [
            "It'll",
            "identify",
            "more",
            "of",
            "the",
            "code.",
        ]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (100, 100),
            (100, 180),
            (180, 250),
            (250, 290),
            (290, 330),
            (330, 420),
        ]

    def test_delete(self) -> None:
        segments = [
            make_seg("It", 0, 50),
            make_seg("kind", 50, 100),
            make_seg("of", 100, 140),
            make_seg("only", 140, 200),
            make_seg("knows", 200, 260),
            make_seg("how", 260, 310),
            make_seg("to", 310, 340),
            make_seg("search", 340, 410),
            make_seg("for", 410, 450),
            make_seg("ASCII", 450, 520),
            make_seg("strings.", 520, 620),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(
            segments, "It only knows how to search for ASCII strings."
        )

        assert [seg.text for seg in rewritten] == [
            "It",
            "only",
            "knows",
            "how",
            "to",
            "search",
            "for",
            "ASCII",
            "strings.",
        ]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 50),
            (140, 200),
            (200, 260),
            (260, 310),
            (310, 340),
            (340, 410),
            (410, 450),
            (450, 520),
            (520, 620),
        ]

    def test_replace_span(self) -> None:
        segments = [
            make_seg("Hit", 0, 90),
            make_seg("that", 90, 150),
            make_seg("later.", 150, 240),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(
            segments, "Get to that later."
        )

        assert [seg.text for seg in rewritten] == ["Get", "to", "that", "later."]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 45),
            (45, 90),
            (90, 150),
            (150, 240),
        ]

    def test_replace_all(self) -> None:
        segments = [
            make_seg("Hello", 0, 100),
            make_seg("world", 100, 200),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(segments, "Fuck you")

        assert [seg.text for seg in rewritten] == ["Fuck", "you"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 200),
        ]

    def test_replace_tail(self) -> None:
        segments = [
            make_seg("haha", 0, 50),
            make_seg("you", 50, 100),
            make_seg("lose", 100, 170),
            make_seg("because", 170, 260),
            make_seg("i'm", 260, 310),
            make_seg("batman", 310, 400),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(segments, "haha you win")

        assert [seg.text for seg in rewritten] == ["haha", "you", "win"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 50),
            (50, 100),
            (100, 400),
        ]

    def test_insert_gap(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 160, 260),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(
            segments, "hello brave new world"
        )

        assert [seg.text for seg in rewritten] == ["hello", "brave", "new", "world"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 130),
            (130, 160),
            (160, 260),
        ]

    def test_insert_end(self) -> None:
        segments = [
            make_seg("hello", 0, 100),
            make_seg("world", 100, 220),
        ]

        rewritten = rewrite_sentence_segments_with_timestamps(
            segments, "hello world again"
        )

        assert [seg.text for seg in rewritten] == ["hello", "world", "again"]
        assert [(seg.start_time, seg.end_time) for seg in rewritten] == [
            (0, 100),
            (100, 220),
            (220, 220),
        ]


class TestMergeOpcodes:
    def test_equal(self) -> None:
        assert merge_edit_opcodes([("equal", 0, 2, 0, 2)]) == [("equal", 0, 2, 0, 2)]

    def test_insert(self) -> None:
        assert merge_edit_opcodes([("insert", 1, 1, 1, 3)]) == [
            ("insert", 1, 1, 1, 3)
        ]

    def test_delete(self) -> None:
        assert merge_edit_opcodes([("delete", 2, 4, 2, 2)]) == [
            ("delete", 2, 4, 2, 2)
        ]

    def test_replace(self) -> None:
        assert merge_edit_opcodes([("replace", 0, 2, 0, 1)]) == [
            ("replace", 0, 2, 0, 1)
        ]

    def test_merge_replace_delete_insert(self) -> None:
        opcodes = [
            ("replace", 0, 1, 0, 1),
            ("delete", 1, 3, 1, 1),
            ("insert", 3, 3, 1, 2),
        ]

        assert merge_edit_opcodes(opcodes) == [("replace", 0, 3, 0, 2)]

    def test_merge_start(self) -> None:
        opcodes = [
            ("insert", 0, 0, 0, 1),
            ("replace", 0, 1, 1, 2),
            ("equal", 1, 2, 2, 3),
        ]

        assert merge_edit_opcodes(opcodes) == [
            ("replace", 0, 1, 0, 2),
            ("equal", 1, 2, 2, 3),
        ]

    def test_merge_end(self) -> None:
        opcodes = [
            ("equal", 0, 1, 0, 1),
            ("delete", 1, 2, 1, 1),
            ("insert", 2, 2, 1, 3),
        ]

        assert merge_edit_opcodes(opcodes) == [
            ("equal", 0, 1, 0, 1),
            ("replace", 1, 2, 1, 3),
        ]

    def test_keep_equal_boundary(self) -> None:
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

    def test_merge_inserts(self) -> None:
        opcodes = [
            ("insert", 1, 1, 1, 2),
            ("insert", 1, 1, 2, 4),
        ]

        assert merge_edit_opcodes(opcodes) == [("insert", 1, 1, 1, 4)]

    def test_merge_deletes(self) -> None:
        opcodes = [
            ("delete", 1, 2, 1, 1),
            ("delete", 2, 4, 1, 1),
        ]

        assert merge_edit_opcodes(opcodes) == [("delete", 1, 4, 1, 1)]
