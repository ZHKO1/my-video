"""Unified LLM client for the application."""

import json
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import json_repair
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
from my_video.core.workspace import format_beijing_time

_global_client: OpenAI | None = None
_client_lock = threading.Lock()


class LLMResponseValidator:
    @staticmethod
    def get_response_id(response: Any) -> str:
        response_id = getattr(response, "id", None)
        if response_id:
            return str(response_id)
        if hasattr(response, "model_dump"):
            try:
                response_dict = response.model_dump()
            except Exception:
                response_dict = None
            if isinstance(response_dict, dict) and response_dict.get("id"):
                return str(response_dict["id"])
        return "-"

    @staticmethod
    def extract_response_text(response: Any) -> tuple[str | None, str]:
        choices = getattr(response, "choices", None)
        if not choices:
            return None, "Response structure error: choices is empty."

        message = getattr(choices[0], "message", None)
        if message is None:
            return None, "Response structure error: message is missing."

        content = getattr(message, "content", None)
        if content is None:
            return None, "Response structure error: content is missing."

        result_text = str(content)
        if not result_text.strip():
            return None, "Response content is empty."

        return result_text, ""

    @staticmethod
    def parse_json_dict(result_text: str) -> tuple[dict[str, Any] | None, str]:
        try:
            parsed_result = json_repair.loads(result_text)
        except Exception as exc:
            return None, f"JSON parse error: {exc}"

        if not isinstance(parsed_result, dict):
            return None, f"JSON structure error: expected dict, got {type(parsed_result).__name__}."

        return parsed_result, ""

    @classmethod
    def ensure_valid_api_response(cls, response: Any) -> None:
        _result_text, error_message = cls.extract_response_text(response)
        if error_message:
            raise ValueError(f"Invalid OpenAI API response: {error_message}")


class LLMLogWriter:
    _log_lock = threading.Lock()

    @staticmethod
    def _load_config() -> dict[str, Any]:
        config, _ = load_toml_config()
        return config or {}

    @classmethod
    def _get_llm_log_path(cls) -> Path:
        config = cls._load_config()
        work_dir = get_work_dir(config) or "."
        return Path(work_dir) / "log" / "llm.log"

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _normalize_log_text(text: Any) -> str:
        if text is None:
            return ""
        return str(text).replace("\\n", "\n")

    @classmethod
    def _get_request_content(cls, messages: list[dict]) -> str:
        if not messages:
            return ""
        return cls._normalize_log_text(messages[-1].get("content", ""))

    @classmethod
    def _get_response_contents(cls, response: Any) -> list[str]:
        if not response or not hasattr(response, "choices") or not response.choices:
            return []

        contents: list[str] = []
        for choice in response.choices:
            message = getattr(choice, "message", None)
            contents.append(cls._normalize_log_text(getattr(message, "content", "")))
        return contents

    @classmethod
    def write(
        cls,
        *,
        status: str,
        messages: list[dict],
        response: Any,
        error: Exception | None,
    ) -> None:
        log_path = cls._get_llm_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"{format_beijing_time()} {LLMResponseValidator.get_response_id(response)} {status}",
            "req:",
            cls._get_request_content(messages),
            "res:",
        ]
        response_contents = cls._get_response_contents(response)
        if not response_contents:
            lines.append("")
        else:
            for index, content in enumerate(response_contents):
                lines.extend([f"choice[{index}]:", content])
        if error is not None:
            lines.extend(["error:", cls._format_exception(error)])

        entry = "\n".join(lines) + "\n\n"

        with cls._log_lock, log_path.open("a", encoding="utf-8") as f:
            f.write(entry)


class LLMRequestClient:
    @staticmethod
    def normalize_base_url(base_url: str) -> str:
        url = base_url.strip()
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not path:
            path = "/v1"

        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    @staticmethod
    def get_client() -> OpenAI:
        global _global_client

        if _global_client is None:
            with _client_lock:
                if _global_client is None:
                    base_url = os.getenv("MY_VIDEO_OPENAI_BASE_URL", "").strip()
                    base_url = LLMRequestClient.normalize_base_url(base_url)
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

    @staticmethod
    def before_sleep_log(_retry_state: RetryCallState) -> None:
        output.warn(
            "Rate Limit Error, sleeping and retrying... Please lower your thread concurrency or use better OpenAI API."
        )


def normalize_base_url(base_url: str) -> str:
    """Normalize API base URL by ensuring /v1 suffix when needed."""
    return LLMRequestClient.normalize_base_url(base_url)


def get_llm_client() -> OpenAI:
    """Get global LLM client instance (thread-safe singleton)."""
    return LLMRequestClient.get_client()


def before_sleep_log(retry_state: RetryCallState) -> None:
    LLMRequestClient.before_sleep_log(retry_state)


def get_response_id(response: Any) -> str:
    return LLMResponseValidator.get_response_id(response)


def extract_response_text(response: Any) -> tuple[str | None, str]:
    return LLMResponseValidator.extract_response_text(response)


def parse_json_dict(result_text: str) -> tuple[dict[str, Any] | None, str]:
    return LLMResponseValidator.parse_json_dict(result_text)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def _request_llm_api(
    messages: list[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    client = LLMRequestClient.get_client()
    return client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        **kwargs,
    )


def _call_llm_api(
    messages: list[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """实际调用 LLM API（带重试）"""
    return _request_llm_api(messages, model, temperature, **kwargs)


# TODO 记得改expire
@memoize(get_llm_cache(), expire=3600000, typed=True)
def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 1,
    **kwargs: Any,
) -> Any:
    """Call LLM API with automatic caching."""
    try:
        response = _call_llm_api(messages, model, temperature, **kwargs)
    except Exception as exc:
        LLMLogWriter.write(status="error", messages=messages, response=None, error=exc)
        raise

    try:
        LLMResponseValidator.ensure_valid_api_response(response)
    except Exception as exc:
        LLMLogWriter.write(
            status="error",
            messages=messages,
            response=response,
            error=exc,
        )
        raise

    LLMLogWriter.write(
        status="success", messages=messages, response=response, error=None
    )
    return response
