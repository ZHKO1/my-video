import subprocess
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from my_video.cli import output
from my_video.core.utils.decorator import skip_fun_if_file_exist
from my_video.core.utils.helper import write_srt


def _ffmpeg_has_encoder(encoder_name: str) -> bool:
    """Check if the current ffmpeg installation supports a given audio encoder."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10
        )
        return encoder_name in result.stdout
    except Exception:
        return False


@skip_fun_if_file_exist(lambda _, audio_path: audio_path)
def convert_video_to_audio(video_path: str, audio_path: str):
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    output.info("Converting to high quality audio with FFmpeg ......")
    if _ffmpeg_has_encoder("libmp3lame"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "32k",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-metadata",
            "encoding=UTF-8",
            str(audio_path),
        ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    output.info(f"Converted <{video_path}> to <{audio_path}> with FFmpeg")


def normalize_audio_volume(
    audio_path: str,
    output_path: str,
    target_db: float = -20.0,
    format: str = "wav",
) -> str:
    audio = AudioSegment.from_file(audio_path)
    change_in_db = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_db)
    normalized_audio.export(output_path, format=format)
    output.info(f"Normalized audio from {audio.dBFS:.1f}dB to {target_db:.1f}dB")
    return output_path


def whisperx_segments_to_srt_entries(
    segments: list[dict[str, Any]],
) -> list[tuple[int, int, int, str]]:
    entries: list[tuple[int, int, int, str]] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start_ms = max(0, round(float(segment["start"]) * 1000))
        end_ms = max(0, round(float(segment["end"]) * 1000))
        entries.append((len(entries) + 1, start_ms, end_ms, text))
    return entries


def save_srt(segments: list[dict], output_path: Path) -> None:
    write_srt(output_path, whisperx_segments_to_srt_entries(segments))
    output.info(f"Subtitle file saved to {output_path!s}")
