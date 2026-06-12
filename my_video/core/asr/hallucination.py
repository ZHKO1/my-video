from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")


@dataclass(frozen=True)
class Token:
    raw: str
    normalized: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class Hit:
    segment_index: int
    start_token: int
    end_token: int
    phrase_tokens: tuple[str, ...]
    repeat_count: int
    start: float | None
    end: float | None
    text_excerpt: str

    @property
    def phrase(self) -> str:
        return " ".join(self.phrase_tokens)

    @property
    def phrase_len(self) -> int:
        return len(self.phrase_tokens)

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token

    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.segment_index, self.start_token, self.end_token, self.phrase_len)


@dataclass(frozen=True)
class ScanResult:
    input_json: Path
    total_segments: int
    total_tokens: int
    hits: list[Hit]

    @property
    def flagged_segments(self) -> int:
        return len({hit.segment_index for hit in self.hits})

    @property
    def total_hits(self) -> int:
        return len(self.hits)

    @property
    def flagged_tokens(self) -> int:
        return sum(hit.token_count for hit in self.hits)

    @property
    def flagged_token_ratio(self) -> float:
        return (self.flagged_tokens / self.total_tokens) if self.total_tokens else 0.0

    @property
    def has_hallucination(self) -> bool:
        return bool(self.hits)

    @property
    def top_phrases(self) -> list[dict[str, Any]]:
        counts = Counter(hit.phrase for hit in self.hits)
        return [
            {"phrase": phrase, "count": count}
            for phrase, count in counts.most_common(10)
        ]

    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_segments": self.total_segments,
            "flagged_segments": self.flagged_segments,
            "total_hits": self.total_hits,
            "flagged_tokens": self.flagged_tokens,
            "flagged_token_ratio": self.flagged_token_ratio,
            "top_phrases": self.top_phrases,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_json": str(self.input_json),
            "summary": self.summary_dict(),
            "hits": [hit_to_dict(hit) for hit in self.hits],
        }


def load_whisperx_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("WhisperX JSON root must be an object")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("WhisperX JSON must contain a 'segments' list")
    return data


def normalize_token(raw: str) -> str | None:
    match = TOKEN_RE.search(raw.lower())
    if match is None:
        return None
    return match.group(0)


def build_segment_tokens(segment: dict[str, Any]) -> list[Token]:
    words = segment.get("words")
    if not isinstance(words, list):
        return []

    tokens: list[Token] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        raw = word.get("word")
        if not isinstance(raw, str):
            continue
        normalized = normalize_token(raw)
        if normalized is None:
            continue
        start = word.get("start")
        end = word.get("end")
        tokens.append(
            Token(
                raw=raw,
                normalized=normalized,
                start=start if isinstance(start, (int, float)) else None,
                end=end if isinstance(end, (int, float)) else None,
            )
        )
    return tokens


def build_excerpt(
    tokens: list[Token], start_token: int, end_token: int, radius: int = 6
) -> str:
    left = max(0, start_token - radius)
    right = min(len(tokens), end_token + radius)
    words = [token.raw for token in tokens[left:right]]
    excerpt = " ".join(words)
    if left > 0:
        excerpt = "... " + excerpt
    if right < len(tokens):
        excerpt = excerpt + " ..."
    return excerpt


def scan_repeated_runs(
    tokens: list[Token],
    segment_index: int,
    min_word_run: int,
    min_phrase_run: int,
    max_phrase_len: int,
) -> list[Hit]:
    hits: list[Hit] = []
    normalized = [token.normalized for token in tokens]
    total = len(normalized)

    for phrase_len in range(1, max_phrase_len + 1):
        min_run = min_word_run if phrase_len == 1 else min_phrase_run
        if total < phrase_len * min_run:
            continue

        i = 0
        while i <= total - phrase_len:
            phrase = tuple(normalized[i : i + phrase_len])
            if len(phrase) < phrase_len:
                break

            repeat_count = 1
            cursor = i + phrase_len
            while (
                cursor + phrase_len <= total
                and tuple(normalized[cursor : cursor + phrase_len]) == phrase
            ):
                repeat_count += 1
                cursor += phrase_len

            if repeat_count >= min_run:
                start_token = i
                end_token = i + repeat_count * phrase_len
                hits.append(
                    Hit(
                        segment_index=segment_index,
                        start_token=start_token,
                        end_token=end_token,
                        phrase_tokens=phrase,
                        repeat_count=repeat_count,
                        start=tokens[start_token].start,
                        end=tokens[end_token - 1].end,
                        text_excerpt=build_excerpt(tokens, start_token, end_token),
                    )
                )
                i = end_token
                continue

            i += 1

    return hits


def overlaps(a: Hit, b: Hit) -> bool:
    if a.segment_index != b.segment_index:
        return False
    return not (a.end_token <= b.start_token or b.end_token <= a.start_token)


def dedupe_hits(hits: list[Hit]) -> list[Hit]:
    ordered = sorted(
        hits,
        key=lambda hit: (
            -hit.token_count,
            -hit.phrase_len,
            -hit.repeat_count,
            hit.segment_index,
            hit.start_token,
        ),
    )
    selected: list[Hit] = []
    for hit in ordered:
        if any(overlaps(hit, existing) for existing in selected):
            continue
        selected.append(hit)
    return sorted(selected, key=Hit.sort_key)


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    return {
        "segment_index": hit.segment_index,
        "start": hit.start,
        "end": hit.end,
        "phrase": hit.phrase,
        "phrase_len": hit.phrase_len,
        "repeat_count": hit.repeat_count,
        "token_count": hit.token_count,
        "text_excerpt": hit.text_excerpt,
    }


def format_time(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:.3f}s"


def format_hallucination_report(result: ScanResult, *, max_examples: int = 30) -> str:
    summary = result.summary_dict()
    lines = [
        "WhisperX hallucination scan",
        f"total_segments: {summary['total_segments']}",
        f"flagged_segments: {summary['flagged_segments']}",
        f"total_hits: {summary['total_hits']}",
        f"flagged_tokens: {summary['flagged_tokens']}",
        f"flagged_token_ratio: {summary['flagged_token_ratio']:.4%}",
        "",
    ]

    top_phrases = summary["top_phrases"]
    if top_phrases:
        lines.append("top_phrases:")
        for item in top_phrases:
            lines.append(f"  - {item['phrase']}: {item['count']}")
        lines.append("")

    if not result.hits:
        lines.append("No repeated-run hallucinations detected.")
        return "\n".join(lines)

    lines.append(f"hits (showing up to {max_examples}):")
    ranked_hits = sorted(
        result.hits,
        key=lambda hit: (
            -hit.token_count,
            -hit.phrase_len,
            -hit.repeat_count,
            hit.segment_index,
            hit.start_token,
        ),
    )
    for hit in ranked_hits[:max_examples]:
        lines.append(
            f"- segment={hit.segment_index} time={format_time(hit.start)}..{format_time(hit.end)} "
            f"phrase={hit.phrase!r} repeat_count={hit.repeat_count} token_count={hit.token_count}"
        )
        lines.append(f"  excerpt: {hit.text_excerpt}")
    return "\n".join(lines)


def scan_whisperx_hallucinations(
    path: Path,
    *,
    min_word_run: int = 5,
    min_phrase_run: int = 4,
    max_phrase_len: int = 4,
) -> ScanResult:
    if min_word_run < 2:
        raise ValueError("min_word_run must be >= 2")
    if min_phrase_run < 2:
        raise ValueError("min_phrase_run must be >= 2")
    if max_phrase_len < 1:
        raise ValueError("max_phrase_len must be >= 1")

    data = load_whisperx_json(path)
    segments = data["segments"]

    all_hits: list[Hit] = []
    total_tokens = 0
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        tokens = build_segment_tokens(segment)
        total_tokens += len(tokens)
        if not tokens:
            continue
        all_hits.extend(
            scan_repeated_runs(
                tokens=tokens,
                segment_index=segment_index,
                min_word_run=min_word_run,
                min_phrase_run=min_phrase_run,
                max_phrase_len=max_phrase_len,
            )
        )

    return ScanResult(
        input_json=path,
        total_segments=len(segments),
        total_tokens=total_tokens,
        hits=dedupe_hits(all_hits),
    )
