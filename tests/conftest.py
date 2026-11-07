import pytest

from my_video.core.llm.client import init_llm_log, reset_llm_log
from my_video.core.utils.cache import close_cache, init_cache


@pytest.fixture(autouse=True)
def workspace_resources(tmp_path):
    init_cache(tmp_path / "cache")
    init_llm_log(tmp_path / "log" / "llm.log")
    yield
    close_cache()
    reset_llm_log()
