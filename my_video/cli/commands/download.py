"""download command — download online video via yt-dlp."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import shutil
import sys

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_str, get_work_dir
from my_video.core.yt_dlp_download import DownloadRequest, download, fetch_video_info
from my_video.core.workspace import (
    create_workspace,
    read_workspace_info,
    resolve_workspace,
)


def _write_info_json(info_path: str | Path, info: dict) -> None:
    path = Path(info_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _should_prompt_for_overwrite(status_info: dict | None) -> bool:
    if not status_info:
        return False
    return status_info.get("stage") == "download" and status_info.get("status") in {"running", "failed"}


def _prompt_overwrite(status_info: dict) -> bool:
    status = status_info.get("status")
    failed_reason = status_info.get("failed_reason")
    last_time = status_info.get("last_time")
    output.warn(f"Workspace already exists with status={status}")
    output.info(f"Last update time: {last_time or ''}")
    if failed_reason:
        output.info(f"Failed reason: {failed_reason}")

    answer = input("Overwrite existing download workspace? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def run(args: Namespace, _config: dict) -> int:
    try:
        config_data = _config
        if config_data is None:
            work_dir = None
            cookie_path = None
            proxy = None
        else:
            work_dir = get_work_dir(config_data)
            cookie_path = get_toml_str(config_data, "download.cookie_path")
            proxy = get_toml_str(config_data, "download.proxy")

        info = fetch_video_info(args.url, cookie_path=cookie_path, proxy=proxy)
        title = info["title"].strip()
        workspace = resolve_workspace(work_dir, title)
        workspace_path = Path(workspace.workspace_path)
        status_info = read_workspace_info(workspace.status_path)

        if status_info and status_info.get("stage") == "download" and status_info.get("status") == "success":
            output.success(f"{title} downloaded: {workspace.workspace_path}/origin/")
            return EXIT.SUCCESS

        if workspace_path.exists() and _should_prompt_for_overwrite(status_info):
            if getattr(args, "force", False):
                output.warn("Overwriting existing download workspace due to --force")
            elif not sys.stdin.isatty():
                output.error("Workspace already exists and requires confirmation; rerun with --force")
                return EXIT.RUNTIME_ERROR
            elif not _prompt_overwrite(status_info):
                output.info("Keeping existing workspace unchanged")
                return EXIT.RUNTIME_ERROR

        if workspace_path.exists():
            shutil.rmtree(workspace_path / "origin", ignore_errors=True)
        else:
            workspace = create_workspace(work_dir, title)

        _write_info_json(workspace.info_path, info)

        download(
            DownloadRequest(
                url=args.url,
                workspace_path=workspace.workspace_path,
                cookie_path=cookie_path,
                proxy=proxy,
            )
        )
        return EXIT.SUCCESS

    except Exception as e:
        output.error(str(e))
        return EXIT.RUNTIME_ERROR
