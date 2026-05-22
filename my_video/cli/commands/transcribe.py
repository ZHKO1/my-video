"""transcribe command — convert audio/video to subtitles via ASR."""

from argparse import Namespace
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_value
from my_video.core.asr_backend.audio_preprocess import (
    convert_video_to_audio,
    normalize_audio_volume,
    process_transcription,
    save_results,
    save_srt,
    split_audio,
)
from my_video.core.asr_backend.demucs_vl import demucs_audio
from my_video.core.asr_backend.whisperx_local import transcribe_audio as transcribe_local_audio
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
    if not video_path.exists():
        output.error(f"Video file not found: {video_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    output_path = str(paths.src_srt)

    try:
        # 1. video to audio
        convert_video_to_audio(video_path, paths)

        # 2. Demucs vocal separation:
        if get_toml_value(config, "transcribe.demucs", False):
            demucs_audio(paths)
            vocal_audio = normalize_audio_volume(str(paths.vocal_audio_file), str(paths.vocal_audio_file), format="mp3")
        else:
            vocal_audio = str(paths.raw_audio_file)

        # 3. Extract audio
        segments = split_audio(str(paths.raw_audio_file))
        
        # 4. Transcribe audio by clips
        all_results = []
        output.info("Transcribing audio with local WhisperX model")

        for index, (start, end) in enumerate(segments, start=1):
            callback(int(index * 100 / max(len(segments), 1)), f"segment {index}/{len(segments)}")
            result = transcribe_local_audio(str(paths.raw_audio_file), vocal_audio, start, end, config)
            all_results.append(result)
       
        # 5. Combine results
        combined_result = {'segments': []}
        for result in all_results:
            combined_result['segments'].extend(result['segments'])
        
        # 6. Process df
        df = process_transcription(combined_result)
        save_results(df, paths)
        save_srt(combined_result["segments"], output_path)
        return EXIT.SUCCESS

    except Exception as e:
        output.error(e)
        return EXIT.RUNTIME_ERROR
