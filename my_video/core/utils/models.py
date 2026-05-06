from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    # base dirs
    output_dir: Path
    log_dir: Path
    audio_dir: Path

    # intermediate files
    cleaned_chunks: Path
    split_by_mark: Path
    split_by_comma: Path
    split_by_connector: Path
    split_by_nlp: Path
    split_by_meaning: Path
    terminology: Path
    translation: Path
    split_sub: Path
    remerged: Path

    # audio files
    audio_task: Path
    raw_audio_file: Path
    vocal_audio_file: Path
    background_audio_file: Path
    audio_refers_dir: Path
    audio_segs_dir: Path
    audio_tmp_dir: Path


def build_output_paths(output_dir: str | Path) -> OutputPaths:
    base = Path(output_dir).expanduser()
    log_dir = base / "log"
    audio_dir = base / "audio"

    return OutputPaths(
        output_dir=base,
        log_dir=log_dir,
        audio_dir=audio_dir,
        cleaned_chunks=log_dir / "cleaned_chunks.xlsx",
        split_by_mark=log_dir / "split_by_mark.txt",
        split_by_comma=log_dir / "split_by_comma.txt",
        split_by_connector=log_dir / "split_by_connector.txt",
        split_by_nlp=log_dir / "split_by_nlp.txt",
        split_by_meaning=log_dir / "split_by_meaning.txt",
        terminology=log_dir / "terminology.json",
        translation=log_dir / "translation_results.xlsx",
        split_sub=log_dir / "translation_results_for_subtitles.xlsx",
        remerged=log_dir / "translation_results_remerged.xlsx",
        audio_task=audio_dir / "tts_tasks.xlsx",
        raw_audio_file=audio_dir / "raw.mp3",
        vocal_audio_file=audio_dir / "vocal.mp3",
        background_audio_file=audio_dir / "background.mp3",
        audio_refers_dir=audio_dir / "refers",
        audio_segs_dir=audio_dir / "segs",
        audio_tmp_dir=audio_dir / "tmp",
    )


__all__ = ["OutputPaths", "build_output_paths"]
