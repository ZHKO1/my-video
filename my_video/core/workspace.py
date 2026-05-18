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
class WorkspaceInfo:
    workspace_path: str
    info_path: str
    status_path: str
    title: str


def sanitize_workspace_name(title: str) -> str:
    """Convert a title into a safe workspace directory name."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", title).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip(" .")
    return cleaned


def create_workspace(work_dir: str | Path, title: str) -> WorkspaceInfo:
    """Create a workspace directory rooted under work_dir."""
    workspace = resolve_workspace(work_dir, title)
    workspace_path = Path(workspace.workspace_path)
    if workspace_path.exists():
        raise FileExistsError(f"Workspace already exists: {workspace_path}")

    workspace_path.mkdir(parents=True, exist_ok=False)
    return workspace


def resolve_workspace(work_dir: str | Path, title: str) -> WorkspaceInfo:
    """Resolve a workspace path from work_dir and title without creating it."""
    workspace_name = sanitize_workspace_name(title)
    if not workspace_name:
        raise ValueError("Video title is empty after sanitizing")

    base_dir = Path(work_dir).expanduser().resolve()
    workspace_path = base_dir / workspace_name
    info_path = workspace_path / "info.json"
    status_path = workspace_path / "status.json"
    return WorkspaceInfo(
        workspace_path=str(workspace_path),
        info_path=str(info_path),
        status_path=str(status_path),
        title=title,
    )


def read_workspace_info(info_path: str | Path) -> dict[str, Any] | None:
    """Read a workspace JSON file if it exists."""
    path = Path(info_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    current = {} if full_update else (read_workspace_info(path) or {})
    payload = dict(updates) if full_update else _merge_dicts(current, updates)
    payload["last_time"] = format_beijing_time()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "WorkspaceInfo",
    "create_workspace",
    "format_beijing_time",
    "read_workspace_info",
    "resolve_workspace",
    "sanitize_workspace_name",
    "update_status_for_workspace",
]
