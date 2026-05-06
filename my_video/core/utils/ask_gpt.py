import json
from pathlib import Path
from threading import Lock

import json_repair
from openai import OpenAI

from my_video.cli import output
from my_video.cli.config import get_toml_str, get_toml_value, get_work_dir, load_toml_config
from my_video.core.utils.decorator import except_handler
from my_video.core.utils.models import build_output_paths

LOCK = Lock()


def _load_config() -> dict:
    config, _ = load_toml_config()
    return config or {}


def _get_gpt_log_dir() -> Path:
    config = _load_config()
    work_dir = get_work_dir(config) or "."
    return build_output_paths(work_dir).output_dir / "gpt_log"


def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = _get_gpt_log_dir() / f"{log_title}.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.exists():
            with file.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(
            {
                "model": model,
                "prompt": prompt,
                "resp_content": resp_content,
                "resp_type": resp_type,
                "resp": resp,
                "message": message,
            }
        )
        with file.open("w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)


def _load_cache(prompt, resp_type, log_title):
    with LOCK:
        file = _get_gpt_log_dir() / f"{log_title}.json"
        if file.exists():
            with file.open("r", encoding="utf-8") as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        return item["resp"]
        return False


def _get_api_base_url(config: dict) -> str | None:
    base_url = get_toml_str(config, "subtitle.translate.api.base_url")
    if not base_url:
        return None
    if "ark" in base_url:
        return "https://ark.cn-beijing.volces.com/api/v3"
    if "v1" not in base_url:
        return base_url.rstrip("/") + "/v1"
    return base_url


@except_handler("GPT request failed", retry=5)
def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default"):
    config = _load_config()
    api_key = get_toml_str(config, "subtitle.translate.api.key")
    if not api_key:
        raise ValueError("API key is not set")

    cached = _load_cache(prompt, resp_type, log_title)
    if cached:
        output.info("Using cached GPT response")
        return cached

    model = get_toml_str(config, "subtitle.translate.api.model", default="gpt-4o-mini") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key, base_url=_get_api_base_url(config))
    response_format = (
        {"type": "json_object"}
        if resp_type == "json"
        and bool(get_toml_value(config, "subtitle.translate.api.llm_support_json", True))
        else None
    )

    resp_raw = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        timeout=300,
    )

    resp_content = resp_raw.choices[0].message.content
    resp = json_repair.loads(resp_content) if resp_type == "json" else resp_content

    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp["status"] != "success":
            _save_cache(
                model,
                prompt,
                resp_content,
                resp_type,
                resp,
                log_title="error",
                message=valid_resp["message"],
            )
            raise ValueError(f"API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
    return resp
