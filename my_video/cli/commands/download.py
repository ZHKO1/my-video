"""download command — download online video via yt-dlp."""

import shutil
import sys
from argparse import Namespace
from pathlib import Path

import tomllib

from my_video.cli import exit_codes as EXIT
from my_video.cli import output


def _resolve_yt_dlp_config_path(args: Namespace) -> Path:
    config_path = getattr(args, "config", None)
    if config_path:
        return Path(config_path)
    return Path("my_video.toml")


def _load_common_config(args: Namespace) -> tuple[list[str], str | None]:
    config_path = _resolve_yt_dlp_config_path(args)
    if not config_path.exists():
        return [], None

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in config file {config_path}: {e}") from e

    download_config = data.get("download")
    if download_config is None:
        return [], None
    if not isinstance(download_config, dict):
        raise ValueError(f"Invalid [download] section in config file {config_path}: expected table")

    yt_dlp_config = download_config.get("yt-dlp")
    if yt_dlp_config is None:
        return [], None
    if not isinstance(yt_dlp_config, dict):
        raise ValueError(
            f"Invalid [download.yt-dlp] section in config file {config_path}: expected table"
        )

    common_config = yt_dlp_config.get("common")
    if common_config is None:
        return [], None
    if not isinstance(common_config, dict):
        raise ValueError(
            f"Invalid [download.yt-dlp.common] section in config file {config_path}: expected table"
        )

    extra_args = common_config.get("extra_args", [])
    if not isinstance(extra_args, list) or any(not isinstance(item, str) for item in extra_args):
        raise ValueError(
            f"Invalid [download.yt-dlp.common].extra_args in config file {config_path}: expected string array"
        )

    output_dir = common_config.get("output_dir")
    if output_dir is not None and not isinstance(output_dir, str):
        raise ValueError(
            f"Invalid [download.yt-dlp.common].output_dir in config file {config_path}: expected string"
        )

    return extra_args, str(Path(output_dir).expanduser()) if output_dir else None


def run(args: Namespace, _config: dict) -> int:
    url = args.url
    quiet = getattr(args, "quiet", False)

    if not shutil.which("yt-dlp"):
        output.error("yt-dlp not found on PATH")
        output.hint("Install: pip install yt-dlp")
        return EXIT.DEPENDENCY_MISSING

    try:
        import subprocess

        extra_args, configured_output_dir = _load_common_config(args)
        out_dir = getattr(args, "output", None) or configured_output_dir or "."

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        base_cmd = ["yt-dlp", "--no-playlist", *extra_args]
        if quiet:
            base_cmd.append("--quiet")

        def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not quiet:
                if result.stdout:
                    sys.stdout.write(result.stdout)
                if result.stderr:
                    sys.stderr.write(result.stderr)
            return result

        video_result = _run_command(
            [
                *base_cmd,
                "--print", "after_move:filepath",
                "-f", "bestvideo+bestaudio/best",
                "-o", f"{out_dir}/%(title)s.%(ext)s",
                url,
            ]
        )
        if video_result.returncode != 0:
            output.error("video+audio download failed")
            return EXIT.RUNTIME_ERROR

        video_lines = [line.strip() for line in video_result.stdout.splitlines() if line.strip()]
        if not video_lines:
            output.error("video+audio filepath not found")
            return EXIT.RUNTIME_ERROR
        video_path = video_lines[-1]

        subtitle_output = str(Path(video_path).with_suffix(".srt"))
        subtitle_result = _run_command(
            [
                *base_cmd,
                "--write-subs",
                "--sub-langs", "en.*",
                "--convert-subs", "srt",
                "--no-embed-subs",
                "--skip-download",
                "-o", f"{subtitle_output}",
                url,
            ]
        )

        subtitle_path = None
        if subtitle_result.returncode != 0:
            output.warn("subtitle download failed, skipping")
        else:
            subtitle_base = Path(subtitle_output)
            matched_subtitles = sorted(subtitle_base.parent.glob(f"{subtitle_base.stem}*.srt"))
            if matched_subtitles:
                subtitle_path = str(matched_subtitles[0])
            else:
                output.warn("subtitle file not found, skipping")

        if not quiet:
            output.success(f"Downloaded to {out_dir}/")
            output.info(f"Video+audio: {video_path}")
            if subtitle_path:
                output.info(f"Subtitle: {subtitle_path}")
        return EXIT.SUCCESS

    except Exception as e:
        output.error(str(e))
        return EXIT.RUNTIME_ERROR
