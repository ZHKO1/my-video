"""Core downloader built around yt-dlp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp

from my_video.cli import output
from my_video.core.workspace import update_status_for_workspace


@dataclass(frozen=True)
class DownloadRequest:
    url: str
    workspace_path: str | Path
    cookie_path: str | Path | None = None
    proxy: str | None = None


@dataclass(frozen=True)
class DownloadResult:
    video_path: str | None
    subtitle_path: str | None
    thumbnail_path: str | None


def _find_named_output(out_dir: str | Path, stem: str, suffixes: tuple[str, ...]) -> str | None:
    """Locate a generated file by fixed basename inside the output directory."""
    base_path = Path(out_dir)

    for suffix in suffixes:
        candidate = base_path / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate.resolve())

    for candidate in sorted(base_path.glob(f"{stem}*")):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def collect_downloaded_paths(out_dir: str | Path) -> DownloadResult:
    """Collect generated output paths using fixed basenames."""
    return DownloadResult(
        video_path=_find_named_output(out_dir, "video", (".mp4", ".mkv", ".webm", ".mov", ".m4a")),
        subtitle_path=_find_named_output(out_dir, "subtitle", (".srt", ".vtt", ".ass", ".lrc")),
        thumbnail_path=_find_named_output(out_dir, "thumb", (".webp", ".jpg", ".jpeg", ".png")),
    )


def fetch_video_info(
    url: str,
    cookie_path: str | Path | None = None,
    proxy: str | None = None,
) -> dict[str, Any]:
    """Fetch complete video info without downloading content."""
    ydl_opts = {
        "noplaylist": True,
        "quiet": True,
        "skip_download": True,
        "remote_components": ["ejs:github"],
    }
    if cookie_path:
        ydl_opts["cookiefile"] = str(Path(cookie_path).expanduser())
    if proxy:
        ydl_opts["proxy"] = proxy

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        raise ValueError("Failed to fetch video title")
    return info


def _reset_status_for_download(status_path: Path, info_path: Path) -> None:
    update_status_for_workspace(
        status_path,
        {
            "stage": "download",
            "status": "running",
            "failed_reason": None,
            "info_path": str(info_path.resolve()),
            "origin": {
                "video_path": None,
                "subtitle_path": None,
                "thumbnail_path": None,
            },
        },
        full_update=True,
    )


def _record_origin_paths(status_path: Path, result: DownloadResult) -> None:
    if result.video_path:
        update_status_for_workspace(status_path, {"origin": {"video_path": result.video_path}})
    if result.subtitle_path:
        update_status_for_workspace(status_path, {"origin": {"subtitle_path": result.subtitle_path}})
    if result.thumbnail_path:
        update_status_for_workspace(status_path, {"origin": {"thumbnail_path": result.thumbnail_path}})


def download(request: DownloadRequest) -> DownloadResult:
    workspace_path = Path(request.workspace_path).expanduser().resolve()
    info_path = workspace_path / "info.json"
    status_path = workspace_path / "status.json"
    origin_path = workspace_path / "origin"

    _reset_status_for_download(status_path, info_path)

    try:
        origin_path.mkdir(parents=False, exist_ok=False)
        ydl_opts = {
            "noplaylist": True,
            "format": "(bestvideo[height<=1080]+bestaudio/best[height<=1080])",
            "writethumbnail": True,
            "writesubtitles": True,
            "writeautomaticsub": False,
            "subtitleslangs": ["en.*"],
            "convertsubtitles": "srt",
            "outtmpl": {
                "default": str(origin_path / "video.%(ext)s"),
                "subtitle": str(origin_path / "subtitle.%(ext)s"),
                "thumbnail": str(origin_path / "thumb.%(ext)s"),
            },
            "postprocessors": [
                {
                    "key": "FFmpegThumbnailsConvertor",
                    "format": "jpg",
                }
            ],
            "remote_components": ["ejs:github"],
        }

        if request.cookie_path:
            ydl_opts["cookiefile"] = str(Path(request.cookie_path).expanduser())
        if request.proxy:
            ydl_opts["proxy"] = request.proxy

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(request.url, download=True)

        result = collect_downloaded_paths(origin_path)
        _record_origin_paths(status_path, result)

        if not result.video_path:
            raise FileNotFoundError("Video file not found after download")
        if not result.subtitle_path:
            output.info("Subtitle file not found, skipping")
        if not result.thumbnail_path:
            output.info("Thumbnail file not found, skipping")

        update_status_for_workspace(
            status_path,
            {
                "status": "success",
                "failed_reason": None,
            },
        )
        output.success(f"Downloaded to {origin_path}/")
        output.info(f"Video+audio: {result.video_path}")
        return result
    except Exception as exc:
        update_status_for_workspace(
            status_path,
            {
                "status": "failed",
                "failed_reason": str(exc),
            },
        )
        output.error(str(exc))
        raise


__all__ = [
    "DownloadRequest",
    "DownloadResult",
    "collect_downloaded_paths",
    "download",
    "fetch_video_info",
]
