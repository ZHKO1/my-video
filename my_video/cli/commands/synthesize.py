"""synthesize command — burn generated subtitles into a video."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import cv2

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_value
from my_video.core.utils.helper import read_json
from my_video.core.workspace import WorkspacePaths, build_workspace_paths

_BEIJING_TZ = ZoneInfo("Asia/Shanghai")

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = "SourceHanSansSC-Regular"
TRANS_FONT_NAME = "SourceHanSansSC-Regular"
FONT_FILE_PATH = str(Path(__file__).resolve().parents[2] / "assets" / "SourceHanSansSC-Regular.otf")
FONT_DIR_PATH = str(Path(FONT_FILE_PATH).parent)

SRC_FONT_COLOR = "&HFFFFFF"
SRC_OUTLINE_COLOR = "&H000000"
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = "&H80000000"
TRANS_FONT_COLOR = "&H00FFFF"
TRANS_OUTLINE_COLOR = "&H000000"
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = "&H33000000"


def check_gpu_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, check=False)
    except Exception:
        return False
    return "h264_nvenc" in result.stdout


def _escape_subtitle_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")


def merge_subtitles_to_video(
    video_file: Path,
    src_srt_path: Path,
    trans_srt_path: Path,
    output_path: Path,
    ffmpeg_gpu: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_srt_path.exists() or not trans_srt_path.exists():
        raise FileNotFoundError(
            f"Subtitle files not found: expected {src_srt_path} and {trans_srt_path}"
        )

    video = cv2.VideoCapture(str(video_file))
    target_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    if target_width <= 0 or target_height <= 0:
        raise RuntimeError(f"Unable to read video resolution from {video_file}")

    output.info(f"Video resolution: {target_width}x{target_height}")

    src_srt = _escape_subtitle_path(src_srt_path)
    trans_srt = _escape_subtitle_path(trans_srt_path)
    font_dir = _escape_subtitle_path(Path(FONT_DIR_PATH))
    src_style = (
        f"FontSize={SRC_FONT_SIZE},FontName={FONT_NAME},"
        f"PrimaryColour={SRC_FONT_COLOR},OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
        f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1"
    )
    trans_style = (
        f"FontSize={TRANS_FONT_SIZE},FontName={TRANS_FONT_NAME},"
        f"PrimaryColour={TRANS_FONT_COLOR},OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
        f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4"
    )
    filter_graph = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
        f"subtitles='{src_srt}':fontsdir='{font_dir}':force_style='{src_style}',"
        f"subtitles='{trans_srt}':fontsdir='{font_dir}':force_style='{trans_style}'"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        str(video_file),
        "-vf",
        filter_graph,
    ]

    if ffmpeg_gpu:
        if check_gpu_available():
            output.info("Using GPU acceleration via h264_nvenc")
            ffmpeg_cmd.extend(["-c:v", "h264_nvenc"])
        else:
            output.warn("synthesize.ffmpeg_gpu is enabled, but h264_nvenc is unavailable; falling back to CPU encoding.")

    ffmpeg_cmd.extend(["-y", str(output_path)])

    output.info("Start merging subtitles to video")
    start_time = time.time()
    process = subprocess.Popen(ffmpeg_cmd)

    try:
        process.wait()
    except Exception as exc:
        if process.poll() is None:
            process.kill()
        raise RuntimeError(f"FFmpeg execution interrupted: {exc}") from exc

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg execution failed with exit code {process.returncode}")

    output.success(f"Subtitle merge complete in {time.time() - start_time:.2f}s -> {output_path}")


def _generate_output_md(paths: WorkspacePaths) -> None:
    if not paths.info_path.exists():
        raise RuntimeError(f"info.json not found: {paths.info_path}")

    try:
        info = read_json(paths.info_path)
    except Exception as exc:
        raise RuntimeError(f"info.json is not valid JSON: {paths.info_path}") from exc

    if info is None:
        raise RuntimeError(f"info.json not found: {paths.info_path}")

    if not isinstance(info, dict):
        raise RuntimeError(f"info.json must contain a JSON object: {paths.info_path}")

    required_fields = ["webpage_url", "title", "uploader", "release_timestamp"]
    missing = [f for f in required_fields if f not in info]
    if missing:
        raise RuntimeError(f"info.json missing required fields: {', '.join(missing)}")

    raw_ts = info["release_timestamp"]
    try:
        timestamp = int(raw_ts)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"release_timestamp is not a valid integer: {raw_ts!r}") from exc

    dt = datetime.fromtimestamp(timestamp, _BEIJING_TZ)
    release_date = f"{dt.year}年{dt.month}月{dt.day}日"

    content = (
        f"视频: {info['webpage_url']}\n"
        f"标题: {info['title']}\n"
        f"作者: {info['uploader']}\n"
        f"原视频投稿时间: {release_date}\n"
    )
    paths.output_md.write_text(content, encoding="utf-8")


def run(args: Namespace, config: dict) -> int:
    workspace_path = Path(args.workspace_path).expanduser()
    if not workspace_path.is_dir():
        output.error(f"Workspace not found: {workspace_path}")
        return EXIT.FILE_NOT_FOUND

    status_path = workspace_path / "status.json"
    status = read_json(status_path)
    if status is None:
        output.error(f"status.json not found: {status_path}")
        return EXIT.FILE_NOT_FOUND

    origin = status.get("origin")
    video_path = origin.get("video_path") if isinstance(origin, dict) else None
    if not video_path:
        output.error(f"origin.video_path missing in status.json: {status_path}")
        return EXIT.RUNTIME_ERROR

    input_path = Path(video_path).expanduser()
    if not input_path.exists():
        output.error(f"Video file not found: {input_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    ffmpeg_gpu = bool(get_toml_value(config, "synthesize.ffmpeg_gpu", False))

    try:
        merge_subtitles_to_video(
            input_path,
            paths.src_srt,
            paths.trans_srt,
            paths.output_mp4,
            ffmpeg_gpu,
        )
        _generate_output_md(paths)
        output.success(f"output.md generated -> {paths.output_md}")
    except Exception as exc:
        output.error(str(exc))
        return EXIT.RUNTIME_ERROR

    return EXIT.SUCCESS
