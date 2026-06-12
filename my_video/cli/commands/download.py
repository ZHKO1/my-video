"""download command — download online video via yt-dlp."""

from __future__ import annotations

import traceback
from argparse import Namespace

from my_video.cli import EXIT, output
from my_video.cli.config import get_toml_str, get_work_dir
from my_video.core.download.yt_dlp import DownloadRequest, download, fetch_video_info
from my_video.core.utils.helper import write_json
from my_video.core.workspace import (
    build_workspace_paths,
    create_workspace,
    update_status_for_workspace,
)


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
        workspace_path = create_workspace(work_dir, title)
        paths = build_workspace_paths(workspace_path)

        write_json(paths.info_path, info)

        # 初始化 status
        update_status_for_workspace(
            paths.status_path,
            {
                "stage": "download",
                "status": "running",
                "failed_reason": None,
                "origin": {
                    "video_path": None,
                    "subtitle_path": None,
                    "thumbnail_path": None,
                },
            },
            full_update=True,
        )

        result = download(
            DownloadRequest(
                url=args.url,
                origin_dir=str(paths.origin_dir),
                cookie_path=cookie_path,
                proxy=proxy,
            )
        )

        # 记录下载结果
        if result.video_path:
            update_status_for_workspace(
                paths.status_path, {"origin": {"video_path": result.video_path}}
            )
        if result.subtitle_path:
            update_status_for_workspace(
                paths.status_path, {"origin": {"subtitle_path": result.subtitle_path}}
            )
        if result.thumbnail_path:
            update_status_for_workspace(
                paths.status_path, {"origin": {"thumbnail_path": result.thumbnail_path}}
            )

        update_status_for_workspace(
            paths.status_path,
            {"status": "success", "failed_reason": None},
        )
        return EXIT.SUCCESS

    except Exception as e:
        if "paths" in locals() and paths.status_path.exists():
            update_status_for_workspace(
                paths.status_path,
                {"status": "failed", "failed_reason": str(e)},
            )
        output.error(str(e))
        output.error(traceback.format_exc())
        return EXIT.RUNTIME_ERROR
