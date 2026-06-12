from pathlib import Path

from my_video.core.utils.decorator import skip_fun_if_file_exist


def test_skip_fun_if_file_exist_skips_when_dynamic_path_exists(tmp_path: Path) -> None:
    target_file = tmp_path / "output.txt"
    target_file.write_text("done", encoding="utf-8")
    calls = []

    @skip_fun_if_file_exist(lambda payload: payload["path"])
    def wrapped(payload):
        calls.append(payload)

    wrapped({"path": target_file})
    assert calls == []


def test_skip_fun_if_file_exist_single_file_missing_executes_function(
    tmp_path: Path,
) -> None:
    target = tmp_path / "missing.txt"

    @skip_fun_if_file_exist(lambda p: p)
    def generate(path: Path) -> str:
        return f"generated {path}"

    result = generate(target)
    assert result == f"generated {target}"


def test_skip_fun_if_file_exist_single_file_existing_skips_execution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("data")

    @skip_fun_if_file_exist(lambda p: p)
    def generate(path: Path) -> str:
        return f"generated {path}"

    result = generate(target)
    assert result is None


def test_skip_fun_if_file_exist_multiple_files_all_missing_executes(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"

    @skip_fun_if_file_exist(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result == "generated"


def test_skip_fun_if_file_exist_multiple_files_partial_existing_executes(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("data")

    @skip_fun_if_file_exist(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result == "generated"


def test_skip_fun_if_file_exist_multiple_files_all_existing_skips_execution(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("data")
    b.write_text("data")

    @skip_fun_if_file_exist(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result is None


def test_skip_fun_if_file_exist_static_string_path(tmp_path: Path) -> None:
    target = tmp_path / "static.txt"
    target.write_text("data")

    @skip_fun_if_file_exist(str(target))
    def generate() -> str:
        return "generated"

    result = generate()
    assert result is None


def test_skip_fun_if_file_exist_path_like_object(tmp_path: Path) -> None:
    target = tmp_path / "pathlike.txt"
    target.write_text("data")

    @skip_fun_if_file_exist(target)
    def generate() -> str:
        return "generated"

    result = generate()
    assert result is None


def test_skip_fun_if_file_exist_preserves_function_metadata() -> None:
    @skip_fun_if_file_exist(lambda x: x)
    def my_func(x: Path) -> str:
        """My docstring."""
        return "done"

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


def test_skip_fun_if_file_exist_lambda_receives_positional_args(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    @skip_fun_if_file_exist(lambda first, second: second)
    def generate(unused: Path, output: Path) -> str:
        return f"generated {output}"

    result = generate(tmp_path / "unused.txt", target)
    assert result == f"generated {target}"


def test_skip_fun_if_file_exist_lambda_receives_keyword_args(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    @skip_fun_if_file_exist(lambda unused, output: output)
    def generate(unused: Path, output: Path) -> str:
        return f"generated {output}"

    result = generate(unused=tmp_path / "unused.txt", output=target)
    assert result == f"generated {target}"
