from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from my_video.core.asr_backend.hallucination import format_hallucination_report, scan_whisperx_hallucinations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect repeated-word/phrase hallucinations in a WhisperX JSON transcript."
    )
    parser.add_argument("input_json", type=Path, help="Path to WhisperX JSON")
    parser.add_argument(
        "--min-word-run",
        type=int,
        default=5,
        help="Minimum consecutive repeats for a single word to be flagged (default: 5)",
    )
    parser.add_argument(
        "--min-phrase-run",
        type=int,
        default=4,
        help="Minimum consecutive repeats for a phrase of 2+ words to be flagged (default: 4)",
    )
    parser.add_argument(
        "--max-phrase-len",
        type=int,
        default=4,
        help="Maximum phrase length in tokens to scan (default: 4)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Maximum number of hits to print in text mode (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be >= 1")

    result = scan_whisperx_hallucinations(
        args.input_json,
        min_word_run=args.min_word_run,
        min_phrase_run=args.min_phrase_run,
        max_phrase_len=args.max_phrase_len,
    )

    if args.json:
        json.dump(result.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    print(format_hallucination_report(result, max_examples=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
