"""transcribe command — convert audio/video to subtitles via ASR."""

from argparse import Namespace
from pathlib import Path
import traceback

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_str, get_toml_value
from my_video.core.asr.audio_preprocess import (
    convert_video_to_audio,
    save_srt,
)
from my_video.core.asr.demucs import demucs_audio
from my_video.core.asr.hallucination import format_hallucination_report, scan_whisperx_hallucinations
from my_video.core.asr.whisperx_local import whisperx_audio
from my_video.core.utils.helper import read_json
from my_video.core.workspace import build_workspace_paths

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
    if not Path(video_path).exists():
        output.error(f"Video file not found: {video_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    if paths.whisperx_json.exists():
        output.success(f"{str(paths.whisperx_json)} is existed")
        return EXIT.SUCCESS

    try:
        input_audio = paths.raw_audio

        # 1. video to audio
        convert_video_to_audio(video_path, str(paths.raw_audio))

        # 2. Demucs vocal separation:
        if get_toml_value(config, "transcribe.demucs", True):
            demucs_audio(paths.raw_audio, paths.vocal_audio)
            input_audio = paths.vocal_audio
    
        # 3. Transcribe audio
        whisper_language = get_toml_str(config, "transcribe.whisperx.language", default="en")
        model_name = get_toml_str(config, "transcribe.whisperx.model", default="large-v3-turbo")
        model_dir = get_toml_str(config, "transcribe.whisperx.model_dir")
        whisperx_audio(input_audio, paths.whisperx_json, whisper_language, model_name, model_dir)

        hallucination_result = scan_whisperx_hallucinations(paths.whisperx_json)
        if hallucination_result.has_hallucination:
            output.warn(format_hallucination_report(hallucination_result, max_examples=5))
            # raise RuntimeError(f"WhisperX hallucination detected in {paths.whisperx_json}")

        whisperx_data = read_json(paths.whisperx_json)
        save_srt(whisperx_data.get("segments", []), str(paths.whisperx_srt))

        # raw_df = extract_words_dataframe(paths.whisperx_json)        
        # cleaned_df = clean_words_dataframe(raw_df)
        
        # segments = cleaned_df.to_dict("records")
        # save_srt(segments, str(paths.transcribe_srt))

        return EXIT.SUCCESS

    except Exception as e:
        output.error(e)
        output.error(traceback.format_exc())
        return EXIT.RUNTIME_ERROR
