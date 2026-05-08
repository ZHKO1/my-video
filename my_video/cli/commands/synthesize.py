"""synthesize command — burn generated subtitles into a video."""

import subprocess
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_value, get_work_dir
from my_video.core.utils.models import OutputPaths, build_output_paths

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


def _default_output_path(input_path: Path, paths: OutputPaths) -> Path:
    return paths.output_dir / "output_sub.mp4"


def merge_subtitles_to_video(
    video_file: Path,
    paths: OutputPaths,
    output_path: Path,
    config: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not paths.src_srt.exists() or not paths.trans_srt.exists():
        raise FileNotFoundError(
            f"Subtitle files not found under {paths.output_dir}: expected {paths.src_srt.name} and {paths.trans_srt.name}"
        )

    video = cv2.VideoCapture(str(video_file))
    target_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    if target_width <= 0 or target_height <= 0:
        raise RuntimeError(f"Unable to read video resolution from {video_file}")

    output.info(f"Video resolution: {target_width}x{target_height}")

    src_srt = _escape_subtitle_path(paths.src_srt)
    trans_srt = _escape_subtitle_path(paths.trans_srt)
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

    ffmpeg_gpu = bool(get_toml_value(config, "synthesize.ffmpeg_gpu", False))
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


def run(args: Namespace, config: dict) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        output.error(f"Input video not found: {input_path}")
        return EXIT.FILE_NOT_FOUND

    work_dir = get_work_dir(config) or "."
    paths = build_output_paths(work_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output else _default_output_path(input_path, paths)
    quiet = getattr(args, "quiet", False)

    try:
        merge_subtitles_to_video(input_path, paths, output_path, config)
    except Exception as exc:
        output.error(str(exc))
        return EXIT.RUNTIME_ERROR

    if quiet:
        print(output_path)
    return EXIT.SUCCESS
