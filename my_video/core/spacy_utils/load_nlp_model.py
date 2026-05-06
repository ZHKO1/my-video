import spacy
from spacy.cli import download

from my_video.cli import output
from my_video.cli.config import get_toml_str, get_toml_value, load_toml_config
from my_video.core.utils import except_handler

DEFAULT_SPACY_MODEL_MAP = {
    "en": "en_core_web_md",
    "zh": "zh_core_web_md",
    "ja": "ja_core_news_md",
    "fr": "fr_core_news_md",
    "de": "de_core_news_md",
    "es": "es_core_news_md",
    "it": "it_core_news_md",
    "ru": "ru_core_news_md",
}


def _load_config() -> dict:
    config, _ = load_toml_config()
    return config or {}


def _get_spacy_model_map() -> dict[str, str]:
    configured = get_toml_value(_load_config(), "spacy_model_map", {})
    if not isinstance(configured, dict):
        return dict(DEFAULT_SPACY_MODEL_MAP)
    return dict(DEFAULT_SPACY_MODEL_MAP) | {
        str(key): str(value) for key, value in configured.items()
    }


def resolve_spacy_language() -> str:
    config = _load_config()
    whisper_language = get_toml_str(config, "transcribe.whisperx.language", default="auto") or "auto"
    if whisper_language == "auto":
        return get_toml_str(config, "transcribe.whisperx.detected_language", default="en") or "en"
    return whisper_language


def get_spacy_model(language: str):
    spacy_model_map = _get_spacy_model_map()
    model = spacy_model_map.get(language.lower(), "en_core_web_md")
    if language.lower() not in spacy_model_map:
        output.warn(f"spaCy model does not support '{language}', using en_core_web_md as fallback")
    return model


@except_handler("Failed to load NLP Spacy model")
def init_nlp():
    language = resolve_spacy_language()
    model = get_spacy_model(language)
    output.info(f"Loading spaCy model: <{model}>")
    try:
        nlp = spacy.load(model)
    except Exception:
        output.warn(f"Downloading spaCy model: {model}")
        output.warn("If the download fails, check your network and try again.")
        download(model)
        nlp = spacy.load(model)
    output.success("spaCy model loaded successfully")
    return nlp
