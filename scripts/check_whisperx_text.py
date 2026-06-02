#!/usr/bin/env python3
"""Check whisperx.json segments for text that doesn't start with uppercase."""

import json
import sys
import argparse
from pathlib import Path


def check_file(filepath: Path) -> list[dict]:
    """Check a single whisperx.json file and return problematic segments."""
    issues = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    for idx, seg in enumerate(segments):
        text = seg.get("text", "")
        if not text:
            issues.append(
                {
                    "index": idx,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": text,
                    "reason": "empty text",
                }
            )
            continue

        stripped = text.lstrip()
        if not stripped:
            issues.append(
                {
                    "index": idx,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": text,
                    "reason": "only whitespace",
                }
            )
            continue

        first_char = stripped[0]
        if not first_char.isupper():
            issues.append(
                {
                    "index": idx,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": text,
                    "reason": f"starts with '{first_char}' (not uppercase)",
                }
            )

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Check whisperx.json segments for text not starting with uppercase."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Path to whisperx.json file(s) or directory(ies) to search recursively",
    )
    args = parser.parse_args()

    json_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            json_files.extend(path.rglob("whisperx.json"))
        else:
            json_files.append(path)

    if not json_files:
        print("No whisperx.json files found.", file=sys.stderr)
        sys.exit(1)

    total_issues = 0
    for json_file in json_files:
        issues = check_file(json_file)
        if issues:
            total_issues += len(issues)
            print(f"\n{json_file}")
            for issue in issues:
                print(
                    f"  [segment {issue['index']}] start={issue['start']:.3f}s end={issue['end']:.3f}s"
                )
                print(f"    text: {json.dumps(issue['text'])}")
                print(f"    reason: {issue['reason']}")

    print(f"\n{'=' * 50}")
    print(f"Checked {len(json_files)} file(s), found {total_issues} issue(s).")
    if total_issues > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
