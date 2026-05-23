from argparse import Namespace
import json
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import download as download_command
from my_video.cli.main import build_parser
from my_video.core.utils.helper import read_json
from my_video.core.workspace import create_workspace, resolve_workspace, sanitize_workspace_name, update_status_for_workspace
from my_video.core.yt_dlp_download import DownloadRequest, DownloadResult, collect_downloaded_paths, download, fetch_video_info


def test_download_parser_accepts_force_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["download", "https://example.com/demo", "--force"])

    assert args.command == "download"
    assert args.url == "https://example.com/demo"
    assert args.force is True


def test_download_command_builds_request_from_config(monkeypatch, tmp_path: Path) -> None:
    captured: list[DownloadRequest] = []

    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: {"title": "Demo Video", "id": "abc123"})

    def fake_download(request: DownloadRequest) -> DownloadResult:
        captured.append(request)
        return DownloadResult(
            video_path=str(tmp_path / "workspace" / "Demo Video" / "origin" / "video.mp4"),
            subtitle_path=str(tmp_path / "workspace" / "Demo Video" / "origin" / "subtitle.srt"),
            thumbnail_path=str(tmp_path / "workspace" / "Demo Video" / "origin" / "thumb.webp"),
        )

    monkeypatch.setattr(download_command, "download", fake_download)

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=False),
        {
            "work_dir": str(tmp_path / "workspace"),
            "download": {
                "cookie_path": str(tmp_path / "cookies.txt"),
                "proxy": "socks5://127.0.0.1:1080/",
            },
        },
    )

    assert result == EXIT.SUCCESS
    assert len(captured) == 1
    assert captured[0].url == "https://example.com/demo"
    assert captured[0].workspace_path == str(tmp_path / "workspace" / "Demo Video")
    assert captured[0].cookie_path == str(tmp_path / "cookies.txt")
    assert captured[0].proxy == "socks5://127.0.0.1:1080/"

    info = json.loads((tmp_path / "workspace" / "Demo Video" / "info.json").read_text(encoding="utf-8"))
    assert info == {"title": "Demo Video", "id": "abc123"}


def test_download_command_returns_error_when_download_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: {"title": "Demo Video"})
    monkeypatch.setattr(
        download_command,
        "download",
        lambda _request: (_ for _ in ()).throw(RuntimeError("Video filepath not found")),
    )

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=False),
        {"work_dir": str(tmp_path / "workspace")},
    )

    assert result == EXIT.RUNTIME_ERROR
    info = json.loads((tmp_path / "workspace" / "Demo Video" / "info.json").read_text(encoding="utf-8"))
    assert info["title"] == "Demo Video"


def test_download_command_returns_success_when_workspace_already_downloaded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: {"title": "Demo Video"})
    workspace_path = create_workspace(tmp_path / "workspace", "Demo Video")
    update_status_for_workspace(
        workspace_path / "status.json",
        {
            "stage": "download",
            "status": "success",
            "failed_reason": None,
            "info_path": str((workspace_path / "info.json").resolve()),
            "origin": {
                "video_path": str((workspace_path / "origin" / "video.mp4").resolve()),
                "subtitle_path": str((workspace_path / "origin" / "subtitle.srt").resolve()),
                "thumbnail_path": str((workspace_path / "origin" / "thumb.jpg").resolve()),
            },
        },
        full_update=True,
    )

    called = False

    def fake_download(_request: DownloadRequest) -> DownloadResult:
        nonlocal called
        called = True
        return DownloadResult(None, None, None)

    monkeypatch.setattr(download_command, "download", fake_download)

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=False),
        {"work_dir": str(tmp_path / "workspace")},
    )

    assert result == EXIT.SUCCESS
    assert called is False


def test_download_command_redownloads_when_workspace_failed_and_force_is_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: {"title": "Demo Video", "id": "new-info"})
    workspace_path = create_workspace(tmp_path / "workspace", "Demo Video")
    origin_path = workspace_path / "origin"
    origin_path.mkdir()
    (origin_path / "stale.txt").write_text("stale", encoding="utf-8")
    update_status_for_workspace(
        workspace_path / "status.json",
        {
            "stage": "download",
            "status": "failed",
            "failed_reason": "old failure",
            "info_path": str((workspace_path / "info.json").resolve()),
            "origin": {
                "video_path": None,
                "subtitle_path": None,
                "thumbnail_path": None,
            },
        },
        full_update=True,
    )

    captured: list[DownloadRequest] = []

    def fake_download(request: DownloadRequest) -> DownloadResult:
        captured.append(request)
        new_origin = Path(request.workspace_path) / "origin"
        new_origin.mkdir()
        return DownloadResult(
            video_path=str(new_origin / "video.mp4"),
            subtitle_path=None,
            thumbnail_path=None,
        )

    monkeypatch.setattr(download_command, "download", fake_download)

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=True),
        {"work_dir": str(tmp_path / "workspace")},
    )

    assert result == EXIT.SUCCESS
    assert len(captured) == 1
    assert captured[0].url == "https://example.com/demo"
    assert captured[0].workspace_path == str(tmp_path / "workspace" / "Demo Video")
    assert captured[0].cookie_path is None
    assert captured[0].proxy is None
    assert not (origin_path / "stale.txt").exists()
    info = json.loads((workspace_path / "info.json").read_text(encoding="utf-8"))
    assert info == {"title": "Demo Video", "id": "new-info"}


def test_download_command_returns_error_without_force_in_non_interactive_failed_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: {"title": "Demo Video"})
    workspace_path = create_workspace(tmp_path / "workspace", "Demo Video")
    update_status_for_workspace(
        workspace_path / "status.json",
        {
            "stage": "download",
            "status": "failed",
            "failed_reason": "old failure",
            "info_path": str((workspace_path / "info.json").resolve()),
            "origin": {
                "video_path": None,
                "subtitle_path": None,
                "thumbnail_path": None,
            },
        },
        full_update=True,
    )
    monkeypatch.setattr(download_command.sys.stdin, "isatty", lambda: False)

    called = False

    def fake_download(_request: DownloadRequest) -> DownloadResult:
        nonlocal called
        called = True
        return DownloadResult(None, None, None)

    monkeypatch.setattr(download_command, "download", fake_download)

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=False),
        {"work_dir": str(tmp_path / "workspace")},
    )

    assert result == EXIT.RUNTIME_ERROR
    assert called is False


def test_collect_downloaded_paths_detects_outputs(tmp_path: Path) -> None:
    (tmp_path / "video.mp4").write_text("video", encoding="utf-8")
    (tmp_path / "subtitle.srt").write_text("subtitle", encoding="utf-8")
    (tmp_path / "thumb.webp").write_text("thumb", encoding="utf-8")

    result = collect_downloaded_paths(tmp_path)

    assert result == DownloadResult(
        video_path=str(tmp_path / "video.mp4"),
        subtitle_path=str(tmp_path / "subtitle.srt"),
        thumbnail_path=str(tmp_path / "thumb.webp"),
    )


def test_core_download_uses_optional_cookie_and_proxy(monkeypatch, tmp_path: Path) -> None:
    captured_opts = {}
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "info.json").write_text("{}", encoding="utf-8")

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured_opts.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            origin_root = Path(captured_opts["outtmpl"]["default"]).parent
            (origin_root / "video.mp4").write_text("video", encoding="utf-8")
            (origin_root / "subtitle.srt").write_text("subtitle", encoding="utf-8")
            (origin_root / "thumb.jpg").write_text("thumb", encoding="utf-8")
            assert url == "https://example.com/demo"
            assert download is True

    monkeypatch.setattr("my_video.core.yt_dlp_download.yt_dlp.YoutubeDL", FakeYoutubeDL)

    result = download(
        DownloadRequest(
            url="https://example.com/demo",
            workspace_path=workspace_path,
            cookie_path=tmp_path / "cookies.txt",
            proxy="socks5://127.0.0.1:1080/",
        )
    )

    assert captured_opts["cookiefile"] == str(tmp_path / "cookies.txt")
    assert captured_opts["proxy"] == "socks5://127.0.0.1:1080/"
    assert captured_opts["postprocessors"] == [
        {"key": "FFmpegSubtitlesConvertor", "format": "srt", "when": "before_dl"},
        {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
    ]
    assert result.video_path == str((workspace_path / "origin" / "video.mp4").resolve())
    assert result.subtitle_path == str((workspace_path / "origin" / "subtitle.srt").resolve())
    assert result.thumbnail_path == str((workspace_path / "origin" / "thumb.jpg").resolve())
    status = json.loads((workspace_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert status["origin"]["video_path"] == str((workspace_path / "origin" / "video.mp4").resolve())


def test_fetch_video_info_uses_optional_cookie_and_proxy(monkeypatch, tmp_path: Path) -> None:
    captured_opts = {}

    class FakeYoutubeDL:
        def __init__(self, opts):
            captured_opts.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            assert url == "https://example.com/demo"
            assert download is False
            return {"title": "Demo Video"}

    monkeypatch.setattr("my_video.core.yt_dlp_download.yt_dlp.YoutubeDL", FakeYoutubeDL)

    info = fetch_video_info(
        "https://example.com/demo",
        cookie_path=tmp_path / "cookies.txt",
        proxy="socks5://127.0.0.1:1080/",
    )

    assert info == {"title": "Demo Video"}
    assert captured_opts["cookiefile"] == str(tmp_path / "cookies.txt")
    assert captured_opts["proxy"] == "socks5://127.0.0.1:1080/"


def test_download_creates_origin_and_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "info.json").write_text("{}", encoding="utf-8")

    class FakeYoutubeDL:
        def __init__(self, _opts):
            self.opts = _opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download):
            origin_root = Path(self.opts["outtmpl"]["default"]).parent
            (origin_root / "video.webm").write_text("video", encoding="utf-8")
            (origin_root / "subtitle.srt").write_text("subtitle", encoding="utf-8")
            (origin_root / "thumb.jpg").write_text("thumb", encoding="utf-8")
            assert download is True

    monkeypatch.setattr("my_video.core.yt_dlp_download.yt_dlp.YoutubeDL", FakeYoutubeDL)

    result = download(DownloadRequest(url="https://example.com/demo", workspace_path=workspace_path))

    assert result == DownloadResult(
        video_path=str((workspace_path / "origin" / "video.webm").resolve()),
        subtitle_path=str((workspace_path / "origin" / "subtitle.srt").resolve()),
        thumbnail_path=str((workspace_path / "origin" / "thumb.jpg").resolve()),
    )
    status = json.loads((workspace_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert status["origin"]["thumbnail_path"] == str((workspace_path / "origin" / "thumb.jpg").resolve())


def test_sanitize_workspace_name_strips_invalid_characters() -> None:
    assert sanitize_workspace_name(' My:/Video?*Name<>|  ') == "My_Video_Name_"


def test_resolve_workspace_returns_existing_workspace_by_title(tmp_path: Path) -> None:
    workspace = create_workspace(tmp_path, "Demo Video")
    info_path = workspace / "info.json"
    info_path.write_text(json.dumps({"title": "Demo Video"}), encoding="utf-8")

    resolved = resolve_workspace(tmp_path, "Demo Video")

    assert resolved == workspace


def test_read_json_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_json(tmp_path / "missing.json") is None
