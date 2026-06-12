"""Configuration helpers for CLI commands."""

import tomllib
from pathlib import Path
from typing import Any


def _resolve_config_path() -> Path:
    return Path("my_video.toml")


def load_toml_config() -> tuple[dict[str, Any], Path] | tuple[None, Path]:
    """Load TOML config file. Returns (None, path) when file does not exist."""
    path = _resolve_config_path()
    if not path.exists():
        return None, path

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in config file {path}: {e}") from e

    return data, path


def get_toml_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Get TOML value by dotted key, e.g. download.yt-dlp.common.extra_args."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return default
        if part not in current:
            return default
        current = current[part]
    return current


def get_toml_str(
    config: dict[str, Any], dotted_key: str, *, default: str | None = None
) -> str | None:
    """Get string value by dotted key."""
    value = get_toml_value(config, dotted_key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid config key '{dotted_key}': expected string")
    return value


def expand_user_path(path: str | None) -> str | None:
    """Expand ~ in path string."""
    return str(Path(path).expanduser()) if path else None


def get_work_dir(config: dict[str, Any], *, default: str | None = ".") -> str | None:
    """Get global work directory from root key `work_dir`."""
    return expand_user_path(get_toml_str(config, "work_dir", default=default))
