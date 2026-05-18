"""transcribe command — convert audio/video to subtitles via ASR."""

from argparse import Namespace
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_value, get_work_dir
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
from my_video.core.utils.models import build_output_paths

def run(args: Namespace, config: dict) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        output.error(f"Input file not found: {input_path}")
        return EXIT.FILE_NOT_FOUND

    work_dir = get_work_dir(config) or "."
    paths = build_output_paths(work_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    out_fmt = "srt"
    output_path = str((paths.output_dir / input_path.name).with_suffix(f".{out_fmt}"))

    verbose = getattr(args, "verbose", False)

    progress = output.ProgressLine(f"Transcribing...").start()

    def callback(pct: int, msg: str) -> None:
        if progress:
            progress.update(pct, f"Transcribing {msg}")

    try:
        video_file = str(input_path)

        # 1. video to audio
        convert_video_to_audio(video_file, paths)

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

        if progress:
            n = len(segments)
            progress.finish(f"Transcription complete -> {output_path} ({n} segment{'' if n == 1 else 's'})")
        return EXIT.SUCCESS

    except Exception as e:
        if progress:
            progress.fail(e)
        else:
            output.error(e)
        if verbose:
            import traceback
            traceback.print_exc()
        return EXIT.RUNTIME_ERROR
