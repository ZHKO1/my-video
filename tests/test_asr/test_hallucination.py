import json
from pathlib import Path

from my_video.core.asr.hallucination import (
    Token,
    format_hallucination_report,
    scan_repeated_runs,
    scan_whisperx_hallucinations,
)

def test_scan_repeated_runs_detects_single_word_hallucination() -> None:
    tokens = [
        Token(raw="alpha", normalized="alpha", start=0.0, end=0.1),
        Token(raw="uh", normalized="uh", start=0.1, end=0.2),
        Token(raw="uh", normalized="uh", start=0.2, end=0.3),
        Token(raw="uh", normalized="uh", start=0.3, end=0.4),
        Token(raw="uh", normalized="uh", start=0.4, end=0.5),
        Token(raw="uh", normalized="uh", start=0.5, end=0.6),
        Token(raw="omega", normalized="omega", start=0.6, end=0.7),
    ]

    hits = scan_repeated_runs(
        tokens=tokens,
        segment_index=3,
        min_word_run=5,
        min_phrase_run=4,
        max_phrase_len=4,
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.segment_index == 3
    assert hit.phrase == "uh"
    assert hit.repeat_count == 5
    assert hit.token_count == 5
    assert hit.start == 0.1
    assert hit.end == 0.6


def test_scan_whisperx_hallucinations_prefers_longer_phrase_and_formats_report(tmp_path: Path) -> None:
    input_json = tmp_path / "whisperx.json"
    input_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "words": [
                            {"word": "you", "start": 0.0, "end": 0.1},
                            {"word": "know", "start": 0.1, "end": 0.2},
                            {"word": "you", "start": 0.2, "end": 0.3},
                            {"word": "know", "start": 0.3, "end": 0.4},
                            {"word": "you", "start": 0.4, "end": 0.5},
                            {"word": "know", "start": 0.5, "end": 0.6},
                            {"word": "you", "start": 0.6, "end": 0.7},
                            {"word": "know", "start": 0.7, "end": 0.8},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = scan_whisperx_hallucinations(input_json)
    report = format_hallucination_report(result, max_examples=1)

    assert result.has_hallucination is True
    assert result.total_hits == 1
    assert result.hits[0].phrase == "you know"
    assert "WhisperX hallucination scan" in report
    assert "flagged_segments: 1" in report
    assert "phrase='you know'" in report

