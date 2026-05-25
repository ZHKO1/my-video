import re

from my_video.cli import output
from my_video.cli.config import get_toml_str, get_toml_value, load_toml_config
from my_video.core.prompts_old import get_subtitle_trim_prompt
from my_video.core.utils import ask_gpt


def _load_config() -> dict:
    config, _ = load_toml_config()
    return config or {}


def _estimate_reading_duration(text: str, target_language: str) -> float:
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin_words = len([part for part in re.split(r"\s+", text.strip()) if part])

    if target_language.startswith(("zh", "ja", "ko")):
        cjk_seconds = cjk_chars / 4.0 if cjk_chars else 0.0
        latin_seconds = latin_words / 2.5 if latin_words else 0.0
        return max(cjk_seconds, latin_seconds)

    return latin_words / 2.5 if latin_words else max(len(text.strip()) / 12.0, 0.0)


def check_len_then_trim(text, duration):
    config = _load_config()
    target_language = get_toml_str(config, "subtitle.target_language", default="zh") or "zh"
    speed_factor_max = float(get_toml_value(config, "subtitle.translate.trim.speed_factor_max", 1.0) or 1.0)
    estimated_duration = _estimate_reading_duration(text, target_language) / max(speed_factor_max, 0.01)

    output.info(f"Subtitle text: {text}")
    output.info(f"Estimated reading duration: {estimated_duration:.2f}s, available: {duration:.2f}s")

    if estimated_duration <= duration:
        return text

    prompt = get_subtitle_trim_prompt(text, duration)

    def valid_trim(response):
        if "result" not in response:
            return {"status": "error", "message": "No result in response"}
        return {"status": "success", "message": ""}

    try:
        response = ask_gpt(prompt, resp_type="json", log_title="sub_trim", valid_def=valid_trim)
        shortened_text = response["result"]
    except Exception:
        output.warn("AI trim failed, falling back to punctuation cleanup")
        shortened_text = re.sub(r"[,.!?;:，。！？；：]", " ", text).strip()

    output.info(f"Trimmed subtitle: {shortened_text}")
    return shortened_text
