"""download command — download online video via yt-dlp."""

import shutil
import sys
from argparse import Namespace
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_str_list, get_work_dir, load_toml_config


def run(args: Namespace, _config: dict) -> int:
    url = args.url
    quiet = getattr(args, "quiet", False)

    if not shutil.which("yt-dlp"):
        output.error("yt-dlp not found on PATH")
        output.hint("Install: pip install yt-dlp")
        return EXIT.DEPENDENCY_MISSING

    try:
        import subprocess

        config_data, _ = load_toml_config()
        if config_data is None:
            extra_args = []
            configured_work_dir = None
        else:
            extra_args = get_toml_str_list(config_data, "download.yt-dlp.common.extra_args", default=[])
            configured_work_dir = get_work_dir(config_data)
        out_dir = configured_work_dir or "."

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
                "--write-thumbnail",
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
