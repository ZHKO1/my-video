"""Unified LLM client for the application."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

import openai
from openai import OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from my_video.cli import output
from my_video.cli.config import get_work_dir, load_toml_config
from my_video.core.utils.cache import get_llm_cache, memoize

_global_client: Optional[OpenAI] = None
_client_lock = threading.Lock()
_log_lock = threading.Lock()



def normalize_base_url(base_url: str) -> str:
    """Normalize API base URL by ensuring /v1 suffix when needed."""
    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if not path:
        path = "/v1"

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return normalized


def get_llm_client() -> OpenAI:
    """Get global LLM client instance (thread-safe singleton)."""
    global _global_client

    if _global_client is None:
        with _client_lock:
            if _global_client is None:
                base_url = os.getenv("MY_VIDEO_OPENAI_BASE_URL", "").strip()
                base_url = normalize_base_url(base_url)
                api_key = os.getenv("MY_VIDEO_OPENAI_API_KEY", "").strip()

                if not base_url or not api_key:
                    raise ValueError(
                        "MY_VIDEO_OPENAI_BASE_URL and MY_VIDEO_OPENAI_API_KEY environment variables must be set"
                    )

                _global_client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )

    return _global_client


def before_sleep_log(retry_state: RetryCallState) -> None:
    output.warn(
        "Rate Limit Error, sleeping and retrying... Please lower your thread concurrency or use better OpenAI API."
    )


def _load_config() -> dict[str, Any]:
    config, _ = load_toml_config()
    return config or {}


def _get_llm_log_path() -> Path:
    config = _load_config()
    work_dir = get_work_dir(config) or "."
    return Path(work_dir) / "log" / "llm.log"


def _serialize_log_value(value: Any) -> str:
    if value is None:
        return "None"
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _write_llm_log(*, status: str, model: str, messages: List[dict], response: Any, error: Exception | None) -> None:
    log_path = _get_llm_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"status={status}",
        f"model={model}",
        f"messages={_serialize_log_value(messages)}",
        f"response={_serialize_log_value(response)}",
    ]
    if error is not None:
        lines.append(f"error={_format_exception(error)}")

    entry = "\n".join(lines) + "\n\n"

    with _log_lock:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def _call_llm_api(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """实际调用 LLM API（带重试）"""
    client = get_llm_client()

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        **kwargs,
    )

    return response

# TODO 记得改expire
@memoize(get_llm_cache(), expire=360000, typed=True)
def call_llm(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """Call LLM API with automatic caching."""
    try:
        response = _call_llm_api(messages, model, temperature, **kwargs)
    except Exception as exc:
        _write_llm_log(status="error", model=model, messages=messages, response=None, error=exc)
        raise

    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and response.choices[0].message.content
    ):
        error = ValueError("Invalid OpenAI API response: empty choices or content")
        _write_llm_log(status="error", model=model, messages=messages, response=response, error=error)
        raise error

    _write_llm_log(status="success", model=model, messages=messages, response=response, error=None)
    return response
