import functools
import os
import time
from collections.abc import Callable
from os import PathLike

from my_video.cli import output

PathResolver = str | PathLike[str] | Callable[..., str | PathLike[str]]


def except_handler(error_msg, retry=0, delay=1, default_return=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    output.error(f"{error_msg}: {e}, retry: {i + 1}/{retry}")
                    if i == retry:
                        if default_return is not None:
                            return default_return
                        raise last_exception
                    time.sleep(delay * (2**i))

        return wrapper

    return decorator


def _resolve_file_path(file_path: PathResolver, *args, **kwargs) -> str:
    if callable(file_path):
        resolved = file_path(*args, **kwargs)
    else:
        resolved = file_path
    return os.fspath(resolved)


def check_file_exists(*file_paths: PathResolver):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            resolved_paths = [_resolve_file_path(fp, *args, **kwargs) for fp in file_paths]

            if all(os.path.exists(p) for p in resolved_paths):
                targets = ", ".join(f"<{p}>" for p in resolved_paths)
                output.warn(f"File {targets} already exists, skip <{func.__name__}> step.")
                return None
            return func(*args, **kwargs)

        return wrapper

    return decorator
