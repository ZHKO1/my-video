""" CLI — AI-powered video captioning from the command line.

Usage:
    my_video <command> [options]

Commands:
    transcribe   Transcribe audio/video to subtitles
    subtitle     Optimize and/or translate subtitle files
    synthesize   Burn subtitles into video
    process      Full pipeline (transcribe → optimize → translate → synthesize)
    download     Download online video (YouTube, Bilibili, etc.)
    config       Manage configuration
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from my_video.cli import exit_codes as EXIT
from my_video.cli.config import load_toml_config

def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add options common to all commands."""
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

def _build_download_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "download",
        help="Download online video (YouTube, Bilibili, etc.)",
        description="Download video from YouTube, Bilibili, and other sites supported by yt-dlp.",
    )
    p.add_argument("url", help="Video URL")
    p.add_argument("--force", action="store_true", help="Overwrite an existing failed or running workspace")
    # _add_common_options(p)
    p.set_defaults(func=_run_download)

def _build_transcribe_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio/video to subtitles",
        description="Convert audio or video files to subtitle files using ASR (Automatic Speech Recognition).",
    )
    p.add_argument("input", help="Audio or video file path")
    p.add_argument("--format", default="srt", help="Output subtitle format")
    # _add_common_options(p)

    p.set_defaults(func=_run_transcribe)

def _build_subtitle_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "subtitle",
        help="Optimize and/or translate subtitle files",
        description="Process subtitle files with a lightweight placeholder pipeline.",
    )
    p.add_argument("input", help="Subtitle file path")
    # _add_common_options(p)

    p.set_defaults(func=_run_subtitle)

def _build_synthesize_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "synthesize",
        help="Burn subtitles into video",
        description="Render a video with subtitles using a lightweight placeholder pipeline.",
    )
    p.add_argument("input", help="Video file path")
    p.add_argument("--output", help="Output video path")
    # _add_common_options(p)

    p.set_defaults(func=_run_synthesize)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my_video",
        description="AI-powered video captioning — transcribe speech, optimize and translate subtitles, "
                    "then burn them into video with customizable styles (ASS or rounded background).",
        epilog="Run 'my_video <command> --help' for details on each command.",
    )
    parser.add_argument("--version", action="version", version=_get_version())

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    _build_download_parser(subparsers)
    _build_transcribe_parser(subparsers)
    _build_subtitle_parser(subparsers)
    _build_synthesize_parser(subparsers)

    return parser

def _get_version() -> str:
    # Read version without importing config.py (avoids side effects)
    try:
        import importlib.metadata
        return f"my_video {importlib.metadata.version('my_video')}"
    except Exception:
        return "my_video (version unknown)"

def _run_download(args: argparse.Namespace) -> int:
    from my_video.cli.commands.download import run
    config = _load_config()
    return run(args, config)

def _run_transcribe(args: argparse.Namespace) -> int:
    from my_video.cli.commands.transcribe import run
    config = _load_config()
    return run(args, config)

def _run_subtitle(args: argparse.Namespace) -> int:
    from my_video.cli.commands.subtitle import run
    config = _load_config()
    return run(args, config)

def _run_synthesize(args: argparse.Namespace) -> int:
    from my_video.cli.commands.synthesize import run
    config = _load_config()
    return run(args, config)


def _load_config() -> dict:
    config, _ = load_toml_config()
    return config or {}

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return EXIT.SUCCESS

    return args.func(args)
