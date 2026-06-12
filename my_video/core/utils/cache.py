"""Disk cache utility for API responses and computation results.

This module provides a simple interface for caching using diskcache.
Can be used by translation, ASR, and other modules that need caching.
"""

import functools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from diskcache import Cache
from diskcache.core import ENOVAL

from my_video.cli import output
from my_video.cli.config import get_work_dir, load_toml_config

# Global cache switch
_cache_enabled = True


def enable_cache() -> None:
    """Enable caching globally."""
    global _cache_enabled
    _cache_enabled = True


def disable_cache() -> None:
    """Disable caching globally."""
    global _cache_enabled
    _cache_enabled = False


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _cache_enabled


# Predefined cache instances for common use cases
_config, _ = load_toml_config()
_work_dir = Path(get_work_dir(_config) or ".")

_llm_cache = Cache(str(_work_dir / "llm_translation"))
_translate_cache = Cache(str(_work_dir / "translate_results"))


def get_llm_cache() -> Cache:
    """Get LLM translation cache instance."""
    return _llm_cache


def get_translate_cache() -> Cache:
    """Get translate cache instance."""
    return _translate_cache


def memoize(cache_instance: Cache, **kwargs):
    """Decorator to cache function results with global switch support.

    Args:
        cache_instance: Cache instance to use (from get_llm_cache(), etc.)
        **kwargs: Arguments passed to cache.memoize() (expire, typed, etc.)

    Returns:
        Decorated function

    Examples:
        @memoize(get_llm_cache(), expire=3600, typed=True)
        def call_api(prompt: str):
            response = client.chat.completions.create(...)
            if not response.choices:
                raise ValueError("Invalid response")  # Exceptions are not cached
            return response
    """
    expire = kwargs.get("expire")
    tag = kwargs.get("tag")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if not _cache_enabled:
                return func(*args, **kw)

            cache_key = generate_cache_key(
                {
                    "func": f"{func.__module__}.{func.__qualname__}",
                    "args": args,
                    "kwargs": tuple(sorted(kw.items())),
                }
            )

            result = cache_instance.get(cache_key, default=ENOVAL)
            if result is not ENOVAL:
                output.info("cache hit!")
                return result

            result = func(*args, **kw)
            cache_instance.set(cache_key, result, expire=expire, tag=tag)
            return result

        return wrapper

    return decorator


def generate_cache_key(data: Any) -> str:
    """Generate cache key from data (supports dataclasses, dicts, lists).

    Args:
        data: Data to generate key from

    Returns:
        SHA256 hash of the data
    """

    def _serialize(obj: Any) -> Any:
        """Recursively serialize object to JSON-serializable format"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)  # type: ignore
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        else:
            return obj

    serialized_data = _serialize(data)
    data_str = json.dumps(serialized_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()
