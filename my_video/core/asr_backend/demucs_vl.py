import gc
import os

from my_video.cli import output
from my_video.core.utils.models import OutputPaths


def demucs_audio(paths: OutputPaths) -> None:
    if os.path.exists(paths.vocal_audio_file) and os.path.exists(paths.background_audio_file):
        output.warn(
            f"{paths.vocal_audio_file} and {paths.background_audio_file} already exist, skipping Demucs processing."
        )
        return

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

    os.makedirs(paths.audio_dir, exist_ok=True)

    output.info("Loading Demucs model: htdemucs")
    model = get_model("htdemucs")
    separator = PreloadedSeparator(model=model)

    output.info("Separating vocals and background audio")
    _, outputs = separator.separate_audio_file(str(paths.raw_audio_file))

    kwargs = {
        "samplerate": model.samplerate,
        "bitrate": 128,
        "preset": 2,
        "clip": "rescale",
        "as_float": False,
        "bits_per_sample": 16,
    }

    save_audio(outputs["vocals"].cpu(), str(paths.vocal_audio_file), **kwargs)
    background = sum(audio for source, audio in outputs.items() if source != "vocals")
    save_audio(background.cpu(), str(paths.background_audio_file), **kwargs)

    del outputs, background, model, separator
    gc.collect()

    output.info("Demucs separation completed")
