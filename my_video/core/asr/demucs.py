import gc
from pathlib import Path

from my_video.cli import output
from my_video.core.asr.audio_preprocess import normalize_audio_volume
from my_video.core.utils.decorator import check_file_exists

@check_file_exists(lambda _, vocal_audio: vocal_audio)
def demucs_audio(raw_audio: Path, vocal_audio: Path) -> None:
    try:
        import torch
        from demucs.api import Separator
        from demucs.apply import BagOfModels
        from demucs.audio import save_audio
        from demucs.pretrained import get_model
        from torch.cuda import is_available as is_cuda_available
    except ImportError as e:
        raise RuntimeError(
            "Demucs is enabled but not installed. Install it first, for example via scripts/install_demucs_uv.sh."
        ) from e

    class PreloadedSeparator(Separator):
        def __init__(self, model: BagOfModels):
            self._model = model
            self._audio_channels = model.audio_channels
            self._samplerate = model.samplerate
            device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            self.update_parameter(
                device=device,
                shifts=1,
                overlap=0.25,
                split=True,
                segment=None,
                jobs=0,
                progress=True,
                callback=None,
                callback_arg=None,
            )

    vocal_audio.parent.mkdir(parents=True, exist_ok=True)

    output.info("Loading Demucs model: htdemucs")
    model = get_model("htdemucs")
    separator = PreloadedSeparator(model=model)

    output.info("Separating vocals and background audio")
    _, outputs = separator.separate_audio_file(str(raw_audio))
    output.info("separate_audio_file completed")

    kwargs = {
        "samplerate": model.samplerate,
        "bitrate": 128,
        "preset": 2,
        "clip": "rescale",
        "as_float": False,
        "bits_per_sample": 16,
    }

    save_audio(outputs["vocals"].cpu(), str(vocal_audio), **kwargs)
    output.info("save_audio completed")

    del outputs, model, separator
    gc.collect()

    output.info("Demucs separation completed")
    normalize_audio_volume(str(vocal_audio), str(vocal_audio), format="wav")

