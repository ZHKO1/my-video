from argparse import Namespace
import json
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import synthesize
from my_video.cli.main import build_parser
from my_video.core.workspace import build_workspace_paths


def test_synthesize_parser_accepts_workspace_path_only() -> None:
    parser = build_parser()
    args = parser.parse_args(["synthesize", "workspace-dir"])
    assert args.command == "synthesize"
    assert args.workspace_path == "workspace-dir"
    assert callable(args.func)


def _setup_workspace(tmp_path: Path, video_exists: bool = True, info: dict | None = None) -> Path:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    input_file = tmp_path / "origin.mp4"
    if video_exists:
        input_file.write_text("x")
    (workspace_path / "status.json").write_text(
        json.dumps({"origin": {"video_path": str(input_file)}}),
        encoding="utf-8",
    )
    if info is not None:
        (workspace_path / "info.json").write_text(
            json.dumps(info),
            encoding="utf-8",
        )
    return workspace_path


def test_synthesize_generates_output_md(monkeypatch, tmp_path: Path) -> None:
    info = {
        "webpage_url": "https://www.youtube.com/watch?v=6vLhgz9qMpM",
        "title": "GODHAND - What Happened?",
        "uploader": "Matt McMuscles",
        "release_timestamp": 1591747200,
    }
    workspace_path = _setup_workspace(tmp_path, info=info)

    monkeypatch.setattr(
        synthesize,
        "merge_subtitles_to_video",
        lambda *args, **kwargs: None,
    )

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.SUCCESS
    assert (workspace_path / "output.md").exists()
    content = (workspace_path / "output.md").read_text(encoding="utf-8")
    expected = (
        "视频: https://www.youtube.com/watch?v=6vLhgz9qMpM\n"
        "标题: GODHAND - What Happened?\n"
        "作者: Matt McMuscles\n"
        "原视频投稿时间: 2020年6月10日\n"
    )
    assert content == expected


def test_synthesize_returns_error_when_info_json_missing(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace_path = _setup_workspace(tmp_path, info=None)
    info_path = workspace_path / "info.json"
    if info_path.exists():
        info_path.unlink()

    monkeypatch.setattr(
        synthesize,
        "merge_subtitles_to_video",
        lambda *args, **kwargs: None,
    )

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})
    stderr = capsys.readouterr().err

    assert result == EXIT.RUNTIME_ERROR
    assert "info.json not found" in stderr


def test_synthesize_returns_error_when_info_json_invalid(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace_path = _setup_workspace(tmp_path, info=None)
    (workspace_path / "info.json").write_text("not json", encoding="utf-8")

    monkeypatch.setattr(
        synthesize,
        "merge_subtitles_to_video",
        lambda *args, **kwargs: None,
    )

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})
    stderr = capsys.readouterr().err

    assert result == EXIT.RUNTIME_ERROR
    assert "info.json" in stderr


def test_synthesize_returns_error_when_info_json_missing_fields(monkeypatch, tmp_path: Path, capsys) -> None:
    info = {"title": "Only Title"}
    workspace_path = _setup_workspace(tmp_path, info=info)

    monkeypatch.setattr(
        synthesize,
        "merge_subtitles_to_video",
        lambda *args, **kwargs: None,
    )

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})
    stderr = capsys.readouterr().err

    assert result == EXIT.RUNTIME_ERROR
    assert "missing required fields" in stderr


def test_synthesize_returns_error_when_release_timestamp_invalid(monkeypatch, tmp_path: Path, capsys) -> None:
    info = {
        "webpage_url": "https://example.com",
        "title": "Title",
        "uploader": "Uploader",
        "release_timestamp": "not-an-int",
    }
    workspace_path = _setup_workspace(tmp_path, info=info)

    monkeypatch.setattr(
        synthesize,
        "merge_subtitles_to_video",
        lambda *args, **kwargs: None,
    )

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})
    stderr = capsys.readouterr().err

    assert result == EXIT.RUNTIME_ERROR
    assert "release_timestamp is not a valid integer" in stderr


def test_synthesize_returns_error_when_merge_subtitles_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    info = {
        "webpage_url": "https://example.com",
        "title": "Title",
        "uploader": "Uploader",
        "release_timestamp": 0,
    }
    workspace_path = _setup_workspace(tmp_path, info=info)

    def raise_error(*args, **kwargs):
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(synthesize, "merge_subtitles_to_video", raise_error)

    result = synthesize.run(Namespace(workspace_path=str(workspace_path)), {})
    stderr = capsys.readouterr().err

    assert result == EXIT.RUNTIME_ERROR
    assert "ffmpeg failed" in stderr
    assert not (workspace_path / "output.md").exists()
