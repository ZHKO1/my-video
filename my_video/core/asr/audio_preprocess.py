import subprocess
from pathlib import Path

from pydub import AudioSegment

from my_video.cli import output
from my_video.core.utils.decorator import skip_fun_if_file_exist


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


def save_srt(segments: list[dict], output_path: Path) -> None:
    def _fmt(ts: float) -> str:
        total_ms = max(0, round(ts * 1000))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    with open(str(output_path), "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            text = segment.get("text", "").strip()
            if not text:
                continue
            f.write(f"{index}\n")
            f.write(f"{_fmt(segment['start'])} --> {_fmt(segment['end'])}\n")
            f.write(f"{text}\n\n")

    output.info(f"Subtitle file saved to {output_path!s}")
