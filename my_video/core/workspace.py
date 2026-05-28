"""Workspace helpers for download pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo



_BEIJING_TZ = ZoneInfo("Asia/Shanghai")
_MISSING = object()


@dataclass(frozen=True)
class WorkspacePaths:
    # base dirs
    base_dir: Path
    transcribe_dir: Path
    subtitle_dir: Path

    origin_dir: Path
    info_path: Path
    status_path: Path

    raw_audio: Path
    vocal_audio: Path
    whisperx_json: Path
    word_timestamps: Path
    cleaned_word_timestamps: Path
    transcribe_srt: Path
    src_srt: Path
    trans_srt: Path
    output_mp4: Path

def build_workspace_paths(workspace_path: str | Path) -> WorkspacePaths:
    base = Path(workspace_path).expanduser().resolve()
    transcribe_dir = base / "transcribe"
    subtitle_dir = base / "subtitle"

    return WorkspacePaths(
        base_dir=base,
        transcribe_dir=transcribe_dir,
        subtitle_dir=subtitle_dir,
        origin_dir=base / "origin",

        info_path=base / "info.json",
        status_path=base / "status.json",
        raw_audio=transcribe_dir / "raw.mp3",
        vocal_audio=transcribe_dir / "vocal.mp3",
        whisperx_json=transcribe_dir / "whisperx.json",
        word_timestamps=transcribe_dir / "word_timestamps.xlsx",
        cleaned_word_timestamps=transcribe_dir / "cleaned_word_timestamps.xlsx",
        transcribe_srt=transcribe_dir / "transcribe.srt",
        src_srt= subtitle_dir / "src.srt",
        trans_srt= subtitle_dir / "trans.srt",
        output_mp4=base/ "output.mp4",

    )


def sanitize_workspace_name(title: str) -> str:
    """Convert a title into a safe workspace directory name."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    return cleaned


def resolve_workspace(work_dir: str | Path, title: str) -> Path:
    """Resolve a workspace path from work_dir and title."""
    workspace_name = sanitize_workspace_name(title)
    if not workspace_name:
        raise ValueError("Video title is empty after sanitizing")

    base_dir = Path(work_dir).expanduser().resolve()
    return base_dir / workspace_name


def create_workspace(work_dir: str | Path, title: str) -> Path:
    """Create a workspace directory rooted under work_dir."""
    workspace_path = resolve_workspace(work_dir, title)
    if workspace_path.exists():
        raise FileExistsError(f"Workspace already exists: {workspace_path}")

    workspace_path.mkdir(parents=True, exist_ok=False)
    return workspace_path


def _merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_dicts(base[key], value)
        else:
            merged[key] = value
    return merged


def format_beijing_time() -> str:
    return datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")


def update_status_for_workspace(
    status_path: str | Path,
    key_or_mapping: str | dict[str, Any],
    value: Any = _MISSING,
    full_update: bool = False,
) -> None:
    """Incrementally update or fully overwrite workspace status.json."""
    if isinstance(key_or_mapping, dict):
        updates = dict(key_or_mapping)
        if isinstance(value, bool) and full_update is False:
            full_update = value
        elif value is not _MISSING:
            raise TypeError("value is only supported when updating a single key")
    elif isinstance(key_or_mapping, str):
        if value is _MISSING:
            raise TypeError("value is required when updating a single key")
        updates = {key_or_mapping: value}
    else:
        raise TypeError("key_or_mapping must be a string key or a dict")

    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from my_video.core.utils.helper import read_json
    current = {} if full_update else (read_json(path) or {})
    payload = dict(updates) if full_update else _merge_dicts(current, updates)
    payload["last_time"] = format_beijing_time()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "WorkspacePaths",
    "build_workspace_paths",
    "create_workspace",
    "format_beijing_time",
    "resolve_workspace",
    "sanitize_workspace_name",
    "update_status_for_workspace",
]
