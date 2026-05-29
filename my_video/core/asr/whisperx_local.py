import functools
import json
import os
from pathlib import Path
import subprocess
import time
import warnings

import torch
import whisperx
from whisperx.audio import SAMPLE_RATE as WHISPER_SAMPLE_RATE
from whisperx.audio import load_audio as whisperx_load_audio

from my_video.cli import output
from my_video.core.utils.decorator import check_file_exists


warnings.filterwarnings("ignore")


# Compatibility shim — applied before WhisperX model loading.
# PyTorch >=2.6 changed torch.load default to weights_only=True, but
# pyannote checkpoints still expect the historical behavior.
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def check_hf_mirror() -> str | None:
    mirrors = {"Official": "huggingface.co", "Mirror": "hf-mirror.com"}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float("inf")

    output.info("Checking HuggingFace mirrors")
    for name, domain in mirrors.items():
        if os.name == "nt":
            cmd = ["ping", "-n", "1", "-w", "3000", domain]
        else:
            cmd = ["ping", "-c", "1", "-W", "3", domain]

        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError:
            output.warn("ping is not available, using default HuggingFace endpoint")
            return None

        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            output.info(f"{name} mirror responded in {response_time:.2f}s")

    if best_time == float("inf"):
        output.warn("All HuggingFace mirrors failed, using default endpoint")
        return None

    output.info(f"Selected HuggingFace endpoint: {fastest_url}")
    return fastest_url

@check_file_exists(lambda _, whisperx_json, *_args, **_kwargs: whisperx_json)
def whisperx_audio(
    audio_file: Path,
    whisperx_json: Path,
    whisper_language: str,
    model_name: str,
    model_dir = str | None,
):
    hf_endpoint = check_hf_mirror()
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
        output.info(
            f"Using WhisperX on {device} (GPU memory={gpu_mem:.2f}GB, batch_size={batch_size}, compute_type={compute_type})"
        )
    else:
        batch_size = 1
        compute_type = "int8"
        output.info(f"Using WhisperX on {device} (batch_size={batch_size}, compute_type={compute_type})")

    model_ref = model_name
    if model_dir:
        candidate = Path(model_dir) / model_name
        if candidate.exists():
            model_ref = str(candidate)

    language = None if whisper_language == "auto" else whisper_language
    output.info("Starting WhisperX")
    model = whisperx.load_model(
        model_ref,
        device,
        compute_type=compute_type,
        language=language,
        vad_options={"vad_onset": 0.500, "vad_offset": 0.363},
        asr_options={"temperatures": [0], "initial_prompt": ""},
        download_root=model_dir,
    )

    full_audio = whisperx_load_audio(str(audio_file), sr=WHISPER_SAMPLE_RATE)

    result = model.transcribe(full_audio, batch_size=batch_size, print_progress=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Align timestamps against the vocal track after raw-audio transcription.
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        full_audio,
        device,
        return_char_alignments=False,
        print_progress=True
    )

    del model_a
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    whisperx_json.parent.mkdir(parents=True, exist_ok=True)
    whisperx_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
