import os, subprocess
from typing import List, Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo

from my_video.core.utils.models import OutputPaths
from my_video.cli import output

def _ffmpeg_has_encoder(encoder_name: str) -> bool:
    """Check if the current ffmpeg installation supports a given audio encoder."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=10
        )
        return encoder_name in result.stdout
    except Exception:
        return False

def convert_video_to_audio(video_file: str, paths: OutputPaths):
    os.makedirs(paths.audio_dir, exist_ok=True)
    if not os.path.exists(paths.raw_audio_file):
        output.info(f"Converting to high quality audio with FFmpeg ......")
        if _ffmpeg_has_encoder('libmp3lame'):
            cmd = [
                'ffmpeg', '-y', '-i', video_file, '-vn',
                '-c:a', 'libmp3lame', '-b:a', '32k',
                '-ar', '16000', '-ac', '1',
                '-metadata', 'encoding=UTF-8', str(paths.raw_audio_file)
            ]
        else:
            # Fallback: conda-forge ffmpeg often lacks libmp3lame.
            # Output as WAV (PCM) which all ffmpeg builds support.
            # Downstream readers (pydub, librosa, whisperX) detect format by
            # file header, not extension, so .mp3 path with WAV content works.
            output.info("libmp3lame not found in ffmpeg, falling back to WAV (PCM) encoding")
            cmd = [
                'ffmpeg', '-y', '-i', video_file, '-vn',
                '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-f', 'wav', str(paths.raw_audio_file)
            ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        output.info(f"Converted <{video_file}> to <{paths.raw_audio_file}> with FFmpeg")


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


def split_audio(
    audio_file: str,
    target_len: float = 30 * 60,
    win: float = 60,
) -> List[Tuple[float, float]]:
    output.info(f"Starting audio segmentation: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    duration = float(mediainfo(audio_file)["duration"])
    if duration <= target_len + win:
        return [(0.0, duration)]

    segments: List[Tuple[float, float]] = []
    pos = 0.0
    safe_margin = 0.5

    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration))
            break

        threshold = pos + target_len
        ws = int((threshold - win) * 1000)
        we = int((threshold + win) * 1000)

        silence_regions = detect_silence(
            audio[ws:we],
            min_silence_len=int(safe_margin * 1000),
            silence_thresh=-30,
        )
        silence_regions = [
            (start / 1000 + (threshold - win), end / 1000 + (threshold - win))
            for start, end in silence_regions
        ]
        valid_regions = [
            (start, end)
            for start, end in silence_regions
            if (end - start) >= (safe_margin * 2)
            and threshold <= start + safe_margin <= threshold + win
        ]

        if valid_regions:
            start, _ = valid_regions[0]
            split_at = start + safe_margin
        else:
            output.warn(
                f"No valid silence regions found for {audio_file} at {threshold:.1f}s, using threshold"
            )
            split_at = threshold

        segments.append((pos, split_at))
        pos = split_at

    output.info(f"Audio split completed: {len(segments)} segment(s)")
    return segments


def process_transcription(result: dict) -> pd.DataFrame:
    all_words: list[dict] = []
    for segment in result["segments"]:
        speaker_id = segment.get("speaker_id")
        for word in segment.get("words", []):
            text = word.get("word", "")
            if len(text) > 30:
                output.warn(f"Detected word longer than 30 characters, skipping: {text}")
                continue
            text = text.replace("»", "").replace("«", "")

            if "start" not in word and "end" not in word:
                if all_words:
                    all_words.append(
                        {
                            "text": text,
                            "start": all_words[-1]["end"],
                            "end": all_words[-1]["end"],
                            "speaker_id": speaker_id,
                        }
                    )
                    continue
                next_word = next(
                    (candidate for candidate in segment.get("words", []) if "start" in candidate and "end" in candidate),
                    None,
                )
                if next_word is None:
                    raise ValueError(f"No timestamp found for word: {word}")
                all_words.append(
                    {
                        "text": text,
                        "start": next_word["start"],
                        "end": next_word["end"],
                        "speaker_id": speaker_id,
                    }
                )
                continue

            all_words.append(
                {
                    "text": text,
                    "start": word.get("start", all_words[-1]["end"] if all_words else 0),
                    "end": word["end"],
                    "speaker_id": speaker_id,
                }
            )

    return pd.DataFrame(all_words)


def save_results(df: pd.DataFrame, paths: OutputPaths) -> None:
    os.makedirs(paths.log_dir, exist_ok=True)

    initial_rows = len(df)
    df = df[df["text"].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        output.info(f"Removed {removed_rows} row(s) with empty text.")

    long_words = df[df["text"].str.len() > 30]
    if not long_words.empty:
        output.warn(f"Detected {len(long_words)} word(s) longer than 30 characters. These will be removed.")
        df = df[df["text"].str.len() <= 30]

    df = df.copy()
    df["text"] = df["text"].apply(lambda text: f'"{text}"')
    df.to_excel(paths.cleaned_chunks, index=False)
    output.info(f"Excel file saved to {paths.cleaned_chunks}")


def save_srt(segments: list[dict], output_path: str) -> None:
    def _fmt(ts: float) -> str:
        total_ms = max(0, round(ts * 1000))
        hours, rem = divmod(total_ms, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    with open(output_path, "w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, start=1):
            text = segment.get("text", "").strip()
            if not text:
                continue
            f.write(f"{index}\n")
            f.write(f"{_fmt(segment['start'])} --> {_fmt(segment['end'])}\n")
            f.write(f"{text}\n\n")

    output.info(f"Subtitle file saved to {output_path}")
