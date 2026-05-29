from pathlib import Path
import subprocess
from typing import List, Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.utils import mediainfo

from my_video.core.utils.decorator import check_file_exists
from my_video.core.utils.helper import read_json
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

@check_file_exists(lambda _, audio_path: audio_path)
def convert_video_to_audio(video_path: str, audio_path: str):
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    output.info("Converting to high quality audio with FFmpeg ......")
    if _ffmpeg_has_encoder('libmp3lame'):
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn',
            '-c:a', 'libmp3lame', '-b:a', '32k',
            '-ar', '16000', '-ac', '1',
            '-metadata', 'encoding=UTF-8', str(audio_path)
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


def split_audio(
    audio_path: str,
    target_len: float = 30 * 60,
    win: float = 60,
) -> List[Tuple[float, float]]:
    output.info(f"Starting audio segmentation: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    duration = float(mediainfo(audio_path)["duration"])
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
                f"No valid silence regions found for {audio_path} at {threshold:.1f}s, using threshold"
            )
            split_at = threshold

        segments.append((pos, split_at))
        pos = split_at

    output.info(f"Audio split completed: {len(segments)} segment(s)")
    return segments


def extract_words_dataframe(result: Path) -> pd.DataFrame:
    loaded_result = read_json(result)
    if loaded_result is None:
        raise FileNotFoundError(f"WhisperX result JSON not found: {result}")
    result = loaded_result

    all_words: list[dict] = []
    for segment in result["segments"]:
        speaker_id = segment.get("speaker_id")
        for word in segment.get("words", []):
            all_words.append(
                {
                    "text": word.get("word", ""),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "speaker_id": speaker_id,
                }
            )

    return pd.DataFrame(all_words, columns=["text", "start", "end", "speaker_id"])


def clean_words_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["text", "start", "end", "speaker_id"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned_df = df.loc[:, required_columns].copy()
    cleaned_df["text"] = cleaned_df["text"].fillna("").astype(str)
    cleaned_df["text"] = cleaned_df["text"].str.replace("»", "", regex=False).str.replace("«", "", regex=False)

    if cleaned_df.empty:
        return cleaned_df.reset_index(drop=True)

    next_valid_timestamp: tuple[float, float] | None = None
    for _, row in cleaned_df.iterrows():
        if pd.notna(row["start"]) and pd.notna(row["end"]):
            next_valid_timestamp = (float(row["start"]), float(row["end"]))
            break

    if next_valid_timestamp is None:
        raise ValueError("No timestamp found in words dataframe")

    previous_end: float | None = None
    normalized_rows: list[dict] = []
    for row in cleaned_df.to_dict("records"):
        start = row["start"]
        end = row["end"]

        has_start = pd.notna(start)
        has_end = pd.notna(end)

        if not has_start and not has_end:
            if previous_end is not None:
                start = previous_end
                end = previous_end
            else:
                start, end = next_valid_timestamp
        elif not has_start:
            end = float(end)
            start = previous_end if previous_end is not None else 0.0
        elif not has_end:
            start = float(start)
            end = start
        else:
            start = float(start)
            end = float(end)

        previous_end = float(end)
        normalized_rows.append(
            {
                "text": row["text"],
                "start": float(start),
                "end": float(end),
                "speaker_id": row["speaker_id"],
            }
        )

    normalized_df = pd.DataFrame(normalized_rows, columns=required_columns)

    initial_rows = len(normalized_df)
    normalized_df = normalized_df[normalized_df["text"].str.len() > 0].copy()
    removed_rows = initial_rows - len(normalized_df)
    if removed_rows > 0:
        output.info(f"Removed {removed_rows} row(s) with empty text.")

    long_words = normalized_df[normalized_df["text"].str.len() > 30]
    if not long_words.empty:
        for text in long_words["text"]:
            output.warn(f"Detected word longer than 30 characters, skipping: {text}")
        normalized_df = normalized_df[normalized_df["text"].str.len() <= 30].copy()

    return normalized_df.reset_index(drop=True)

@check_file_exists(lambda _, output_path: output_path)
def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    output.info(f"Excel file saved to {output_path}")


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

    output.info(f"Subtitle file saved to {str(output_path)}")
