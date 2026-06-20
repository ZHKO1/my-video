from __future__ import annotations

import itertools
from typing import Literal

from rapidfuzz.distance import Levenshtein

from my_video.core.asr.asr_data import SubtitleSegment
from my_video.core.utils.helper import text_tokens, token_bases

DiffMode = Literal["strict", "relaxed"]
Opcode = tuple[str, int, int, int, int]


def build_token_opcodes(
    reference_tokens: list[str],
    candidate_tokens: list[str],
    *,
    mode: DiffMode = "strict",
) -> list[Opcode]:
    if mode == "strict":
        return Levenshtein.opcodes(reference_tokens, candidate_tokens)
    if mode == "relaxed":
        return Levenshtein.opcodes(
            token_bases(reference_tokens),
            token_bases(candidate_tokens),
        )
    raise ValueError(f"Unsupported diff mode: {mode}")


def merge_edit_opcodes(opcodes: list[Opcode]) -> list[Opcode]:
    merged: list[Opcode] = []
    index = 0
    while index < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[index]
        if tag == "equal":
            merged.append((tag, i1, i2, j1, j2))
            index += 1
            continue

        merged_i1, merged_i2 = i1, i2
        merged_j1, merged_j2 = j1, j2
        index += 1
        while index < len(opcodes) and opcodes[index][0] != "equal":
            _, _, next_i2, _, next_j2 = opcodes[index]
            merged_i2 = next_i2
            merged_j2 = next_j2
            index += 1

        if merged_i1 == merged_i2:
            merged_tag = "insert"
        elif merged_j1 == merged_j2:
            merged_tag = "delete"
        else:
            merged_tag = "replace"
        merged.append((merged_tag, merged_i1, merged_i2, merged_j1, merged_j2))

    return merged


def render_inline_diff(
    reference_text: str,
    candidate_text: str,
) -> str:
    reference_tokens = text_tokens(reference_text)
    candidate_tokens = text_tokens(candidate_text)
    opcodes = merge_edit_opcodes(
        build_token_opcodes(reference_tokens, candidate_tokens, mode="strict")
    )

    rendered: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            rendered.extend(candidate_tokens[j1:j2])
            continue

        rendered.extend(
            _render_edit_markers(
                reference_tokens[i1:i2],
                candidate_tokens[j1:j2],
            )
        )

    return " ".join(rendered)


def rewrite_segments_with_timestamps(
    original_segments: list[SubtitleSegment],
    target_text: str,
    *,
    mode: DiffMode = "strict",
    anchor_mode: DiffMode = "relaxed",
) -> list[SubtitleSegment]:
    original_tokens = [segment.text.strip() for segment in original_segments]
    target_tokens = text_tokens(target_text)
    segment_overrides: dict[int, tuple[int, int]] = {}
    opcodes = merge_edit_opcodes(
        build_token_opcodes(
            original_tokens,
            target_tokens,
            mode=anchor_mode or mode,
        )
    )

    rewritten_segments: list[SubtitleSegment] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for old_index, new_index in zip(range(i1, i2), range(j1, j2), strict=True):
                start_time, end_time = segment_overrides.get(
                    old_index,
                    (
                        original_segments[old_index].start_time,
                        original_segments[old_index].end_time,
                    ),
                )
                rewritten_segments.append(
                    SubtitleSegment(
                        text=target_tokens[new_index],
                        start_time=start_time,
                        end_time=end_time,
                    )
                )
            continue

        if tag == "delete":
            continue

        if tag == "insert":
            rewritten_segments.extend(
                _build_insert_segments(
                    original_segments=original_segments,
                    target_tokens=target_tokens,
                    insert_at=i1,
                    token_start=j1,
                    token_end=j2,
                    segment_overrides=segment_overrides,
                )
            )
            continue

        rewritten_segments.extend(
            _build_replace_segments(
                original_segments=original_segments,
                target_tokens=target_tokens,
                original_start=i1,
                original_end=i2,
                token_start=j1,
                token_end=j2,
            )
        )

    return rewritten_segments


def _build_insert_segments(
    *,
    original_segments: list[SubtitleSegment],
    target_tokens: list[str],
    insert_at: int,
    token_start: int,
    token_end: int,
    segment_overrides: dict[int, tuple[int, int]],
) -> list[SubtitleSegment]:
    insert_count = token_end - token_start
    if insert_count <= 0:
        return []

    left_anchor = original_segments[insert_at - 1] if insert_at > 0 else None
    right_anchor = (
        original_segments[insert_at] if insert_at < len(original_segments) else None
    )

    if left_anchor and right_anchor and right_anchor.start_time > left_anchor.end_time:
        time_ranges = _split_time_range(
            left_anchor.end_time, right_anchor.start_time, insert_count
        )
    elif right_anchor is not None:
        time_ranges = _split_time_range(
            right_anchor.start_time, right_anchor.end_time, insert_count + 1
        )
        segment_overrides[insert_at] = time_ranges[-1]
        time_ranges = time_ranges[:-1]
    elif left_anchor is not None:
        time_ranges = _split_time_range(
            left_anchor.start_time, left_anchor.end_time, insert_count + 1
        )
        segment_overrides[insert_at - 1] = time_ranges[0]
        time_ranges = time_ranges[1:]
    else:
        time_ranges = [(0, 0)] * insert_count

    return [
        SubtitleSegment(
            text=target_tokens[token_index],
            start_time=start_time,
            end_time=end_time,
        )
        for token_index, (start_time, end_time) in zip(
            range(token_start, token_end), time_ranges, strict=True
        )
    ]


def _build_replace_segments(
    *,
    original_segments: list[SubtitleSegment],
    target_tokens: list[str],
    original_start: int,
    original_end: int,
    token_start: int,
    token_end: int,
) -> list[SubtitleSegment]:
    old_count = original_end - original_start
    new_count = token_end - token_start

    if new_count <= 0:
        return []

    if old_count == new_count and old_count > 0:
        return [
            _copy_segment_with_text(
                original_segments[old_index], target_tokens[token_index]
            )
            for old_index, token_index in zip(
                range(original_start, original_end),
                range(token_start, token_end),
                strict=True,
            )
        ]

    if old_count <= 0:
        return _build_insert_segments(
            original_segments=original_segments,
            target_tokens=target_tokens,
            insert_at=original_start,
            token_start=token_start,
            token_end=token_end,
        )

    time_ranges = _split_time_range(
        original_segments[original_start].start_time,
        original_segments[original_end - 1].end_time,
        new_count,
    )
    return [
        SubtitleSegment(
            text=target_tokens[token_index],
            start_time=start_time,
            end_time=end_time,
        )
        for token_index, (start_time, end_time) in zip(
            range(token_start, token_end), time_ranges, strict=True
        )
    ]


def _copy_segment_with_text(segment: SubtitleSegment, text: str) -> SubtitleSegment:
    return SubtitleSegment(
        text=text,
        start_time=segment.start_time,
        end_time=segment.end_time,
    )


def _split_time_range(
    start_time: int, end_time: int, count: int
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    if count == 1:
        return [(start_time, end_time)]

    total = end_time - start_time
    points = [start_time + (total * index) // count for index in range(count + 1)]
    return list(itertools.pairwise(points))


def _render_edit_markers(old_tokens: list[str], new_tokens: list[str]) -> list[str]:
    pair_count = min(len(old_tokens), len(new_tokens))
    rendered = [
        f"【{old_tokens[index]}/{new_tokens[index]}】" for index in range(pair_count)
    ]

    if len(old_tokens) > pair_count:
        rendered.append(f"【{' '.join(old_tokens[pair_count:])}/∅】")
    if len(new_tokens) > pair_count:
        rendered.append(f"【∅/{' '.join(new_tokens[pair_count:])}】")
    return rendered
