import functools
import logging
import os
import time
from collections.abc import Callable
from os import PathLike

logger = logging.getLogger(__name__)


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
                    if i == retry:
                        if default_return is not None:
                            logger.warning(
                                "%s: %s; attempts exhausted (%s/%s), using default result",
                                error_msg,
                                e,
                                i + 1,
                                retry + 1,
                            )
                            return default_return
                        raise last_exception from None
                    logger.warning(
                        "%s: %s; attempt %s/%s failed, retrying in %s seconds",
                        error_msg,
                        e,
                        i + 1,
                        retry + 1,
                        delay * (2**i),
                    )
                    time.sleep(delay * (2**i))

        return wrapper

    return decorator


def _resolve_file_path(file_path: PathResolver, *args, **kwargs) -> str:
    resolved = file_path(*args, **kwargs) if callable(file_path) else file_path
    return os.fspath(resolved)


def skip_fun_if_file_exist(*file_paths: PathResolver):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            resolved_paths = [
                _resolve_file_path(fp, *args, **kwargs) for fp in file_paths
            ]

            if all(os.path.exists(p) for p in resolved_paths):
                targets = ", ".join(f"<{p}>" for p in resolved_paths)
                logger.info(
                    f"File {targets} already exists, skip <{func.__name__}> step."
                )
                return None
            return func(*args, **kwargs)

        return wrapper

    return decorator
