"""Workspace helpers for download pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from my_video.core.utils.helper import read_json


_BEIJING_TZ = ZoneInfo("Asia/Shanghai")
_MISSING = object()


@dataclass(frozen=True)
class WorkspacePaths:
    # base dirs
    base_dir: Path
    log_dir: Path
    audio_dir: Path

    # intermediate files
    cleaned_chunks: Path
    split_by_mark: Path
    split_by_comma: Path
    split_by_connector: Path
    split_by_nlp: Path
    split_by_meaning: Path
    terminology: Path
    translation: Path
    split_sub: Path
    remerged: Path
    src_srt: Path
    trans_srt: Path
    src_trans_srt: Path
    trans_src_srt: Path
    src_subs_for_audio_srt: Path
    trans_subs_for_audio_srt: Path

    # audio files
    audio_task: Path
    raw_audio_file: Path
    vocal_audio_file: Path
    background_audio_file: Path


def build_workspace_paths(workspace_path: str | Path) -> WorkspacePaths:
    base = Path(workspace_path).expanduser().resolve()
    log_dir = base / "log"
    audio_dir = base / "audio"

    return WorkspacePaths(
        base_dir=base,
        log_dir=log_dir,
        audio_dir=audio_dir,
        cleaned_chunks=log_dir / "cleaned_chunks.xlsx",
        split_by_mark=log_dir / "split_by_mark.txt",
        split_by_comma=log_dir / "split_by_comma.txt",
        split_by_connector=log_dir / "split_by_connector.txt",
        split_by_nlp=log_dir / "split_by_nlp.txt",
        split_by_meaning=log_dir / "split_by_meaning.txt",
        terminology=log_dir / "terminology.json",
        translation=log_dir / "translation_results.xlsx",
        split_sub=log_dir / "translation_results_for_subtitles.xlsx",
        remerged=log_dir / "translation_results_remerged.xlsx",
        src_srt=base / "src.srt",
        trans_srt=base / "trans.srt",
        src_trans_srt=base / "src_trans.srt",
        trans_src_srt=base / "trans_src.srt",
        src_subs_for_audio_srt=audio_dir / "src_subs_for_audio.srt",
        trans_subs_for_audio_srt=audio_dir / "trans_subs_for_audio.srt",
        audio_task=audio_dir / "tts_tasks.xlsx",
        raw_audio_file=audio_dir / "raw.mp3",
        vocal_audio_file=audio_dir / "vocal.mp3",
        background_audio_file=audio_dir / "background.mp3",
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
