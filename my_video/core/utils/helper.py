from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_SPLIT_PATTERN = re.compile(r"\s+")
WHITESPACE_PATTERN = re.compile(r"\s+")
TRAILING_PUNCTUATION_PATTERN = re.compile(r"[^\w]+$")
EDGE_PUNCTUATION_PATTERN = re.compile(r"^[^\w]+|[^\w]+$")


@dataclass(frozen=True)
class TokenParts:
    original: str
    base: str
    trailing_punctuation: str


def read_json(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_json_source(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source

    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8"))

    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(source)

        path = Path(source)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))

    return json.loads(source)


def write_json_source(
    payload: dict[str, Any],
    save_path: str | Path | None = None,
) -> str:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if save_path is not None:
        file_path = Path(save_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    return content


def split_tokens(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [token for token in TOKEN_SPLIT_PATTERN.split(text.strip()) if token]


def compact_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def text_tokens(text: str) -> list[str]:
    return split_tokens(compact_whitespace(text))


def split_token_parts(token: str) -> TokenParts:
    stripped = token.strip()
    trailing_punctuation_match = TRAILING_PUNCTUATION_PATTERN.search(stripped)
    trailing_punctuation = (
        trailing_punctuation_match.group(0) if trailing_punctuation_match else ""
    )
    base = (
        stripped[: len(stripped) - len(trailing_punctuation)]
        if trailing_punctuation
        else stripped
    )
    return TokenParts(
        original=stripped,
        base=base,
        trailing_punctuation=trailing_punctuation,
    )


# 去掉标点符号，全小写
def text_bases(text: str) -> list[str]:
    return token_bases(split_tokens(text))


def token_bases(tokens: list[str]) -> list[str]:
    return [
        EDGE_PUNCTUATION_PATTERN.sub("", split_token_parts(token).base).lower()
        for token in tokens
    ]
