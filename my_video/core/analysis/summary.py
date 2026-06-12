"""Generate and validate subtitle summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import json_repair

from my_video.core.llm import call_llm
from my_video.core.prompts import get_prompt


def _validate_summary_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Summary response must be a JSON object")

    summary = payload.get("summary")
    terms = payload.get("terms")

    if not isinstance(summary, str):
        raise ValueError("Summary response must contain a string field 'summary'")
    if not isinstance(terms, list):
        raise ValueError("Summary response must contain an array field 'terms'")

    normalized_terms: list[dict[str, str]] = []
    for index, term in enumerate(terms):
        if not isinstance(term, dict):
            raise ValueError(f"Term at index {index} must be an object")
        src = term.get("src")
        tgt = term.get("tgt")
        if not isinstance(src, str) or not isinstance(tgt, str):
            raise ValueError(
                f"Term at index {index} must contain string fields 'src' and 'tgt'"
            )
        normalized_terms.append({"src": src, "tgt": tgt})

    return {
        "summary": summary,
        "terms": normalized_terms,
    }


def get_summary(txt_path: str | Path, model: str = "deepseek-v4-pro") -> dict[str, Any]:
    subtitle_text = Path(txt_path).read_text(encoding="utf-8")
    response = call_llm(
        messages=[
            {"role": "system", "content": get_prompt("analysis/summary")},
            {"role": "user", "content": subtitle_text},
        ],
        model=model,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Summary response is empty")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = json_repair.loads(content)

    return _validate_summary_payload(payload)
