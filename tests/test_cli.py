from argparse import Namespace
import json
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import download as download_command
from my_video.cli.main import build_parser
from my_video.core.utils.helper import read_json
from my_video.core.utils.models import OutputPaths, build_output_paths
from my_video.core.utils.decorator import check_file_exists
from my_video.cli.commands import subtitle, synthesize, transcribe
from my_video.core.yt_dlp_download import DownloadRequest, DownloadResult, collect_downloaded_paths, download, fetch_video_info
from my_video.core.workspace import (
    WorkspacePaths,
    build_workspace_paths,
    create_workspace,
    resolve_workspace,
    sanitize_workspace_name,
    update_status_for_workspace,
)


def test_build_workspace_paths_matches_compat_builder(tmp_path: Path) -> None:
    workspace_input = tmp_path / ".." / tmp_path.name / "workspace"
    new_paths = build_workspace_paths(workspace_input)
    compat_paths = build_output_paths(workspace_input)

    assert isinstance(new_paths, WorkspacePaths)
    assert new_paths == compat_paths
    assert new_paths.output_dir == workspace_input.expanduser().resolve()
    assert new_paths.src_srt == new_paths.output_dir / "src.srt"
    assert new_paths.audio_dir == new_paths.output_dir / "audio"


def test_output_paths_compatibility_aliases() -> None:
    assert OutputPaths is WorkspacePaths
    assert build_output_paths is build_workspace_paths


def test_transcribe_parser_accepts_workspace_path_only() -> None:
    parser = build_parser()
    args = parser.parse_args(["transcribe", "workspace-dir"])
    assert args.command == "transcribe"
    assert args.workspace_path == "workspace-dir"
    assert callable(args.func)

    try:
        parser.parse_args(["transcribe", "workspace-dir", "--format", "srt"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected --format to be rejected")

    try:
        parser.parse_args(["transcribe", "workspace-dir", "--output", "somewhere"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected --output to be rejected")


def test_transcribe_uses_workspace_and_status_video(monkeypatch, tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    input_file = tmp_path / "origin.mp4"
    input_file.write_text("x")
    (workspace_path / "status.json").write_text(
        json.dumps({"origin": {"video_path": str(input_file)}}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    real_builder = build_workspace_paths

    def fake_build_workspace_paths(base_dir: str | Path):
        captured["workspace_path"] = Path(base_dir)
        return real_builder(base_dir)

    monkeypatch.setattr(transcribe, "build_workspace_paths", fake_build_workspace_paths)
    monkeypatch.setattr(transcribe, "convert_video_to_audio", lambda video_file, paths: captured.setdefault("video_file", video_file))
    monkeypatch.setattr(transcribe, "split_audio", lambda _audio: [(0.0, 1.0)])
    monkeypatch.setattr(
        transcribe,
        "transcribe_local_audio",
        lambda raw_audio, vocal_audio, start, end, _config: {
            "segments": [{"start": start, "end": end, "text": "hello"}],
        },
    )
    monkeypatch.setattr(transcribe, "process_transcription", lambda result: {"segments": result["segments"]})
    monkeypatch.setattr(transcribe, "save_results", lambda df, paths: captured.setdefault("saved_results", (df, paths.output_dir)))
    monkeypatch.setattr(transcribe, "save_srt", lambda segments, output_path: captured.setdefault("save_srt", (segments, output_path)))
    monkeypatch.setattr(
        transcribe,
        "normalize_audio_volume",
        lambda input_path, _output_path, format="mp3": input_path,
        raising=False,
    )

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.SUCCESS
    assert captured["workspace_path"] == workspace_path
    assert captured["video_file"] == str(input_file)
    assert captured["save_srt"] == (
        [{"start": 0.0, "end": 1.0, "text": "hello"}],
        str(workspace_path.resolve() / "src.srt"),
    )


def test_transcribe_returns_error_when_workspace_missing(tmp_path: Path) -> None:
    result = transcribe.run(Namespace(workspace_path=str(tmp_path / "missing")), {})
    assert result == EXIT.FILE_NOT_FOUND


def test_transcribe_returns_error_when_status_missing(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.FILE_NOT_FOUND


def test_transcribe_returns_error_when_origin_video_missing_in_status(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "status.json").write_text(json.dumps({"origin": {}}), encoding="utf-8")

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.RUNTIME_ERROR


def test_transcribe_returns_error_when_origin_video_file_missing(tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    missing_video = tmp_path / "missing.mp4"
    (workspace_path / "status.json").write_text(
        json.dumps({"origin": {"video_path": str(missing_video)}}),
        encoding="utf-8",
    )

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.FILE_NOT_FOUND


def test_subtitle_parser_accepts_input_and_common_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["subtitle", "demo.srt"])

    assert args.command == "subtitle"
    assert args.input == "demo.srt"
    assert callable(args.func)


def test_subtitle_returns_success_for_existing_input(tmp_path: Path) -> None:
    input_file = tmp_path / "demo.srt"
    input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    def fake_split_by_spacy(paths):
        paths.split_by_nlp.write_text("ok", encoding="utf-8")

    subtitle.split_by_spacy = fake_split_by_spacy
    subtitle.translate_all = lambda *_args, **_kwargs: None
    subtitle.align_timestamp_main = lambda *_args, **_kwargs: None
    result = subtitle.run(
        Namespace(input=str(input_file)),
        {"work_dir": str(tmp_path / "work-dir")},
    )

    assert result == EXIT.SUCCESS


def test_subtitle_uses_work_dir_from_config(tmp_path: Path) -> None:
    input_file = tmp_path / "demo.srt"
    input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    captured = []

    def fake_split_by_spacy(paths):
        captured.append(paths)

    subtitle.split_by_spacy = fake_split_by_spacy
    subtitle.translate_all = lambda *_args, **_kwargs: None
    subtitle.align_timestamp_main = lambda *_args, **_kwargs: None

    result = subtitle.run(
        Namespace(input=str(input_file)),
        {"work_dir": str(tmp_path / "subtitle-work")},
    )

    assert result == EXIT.SUCCESS
    assert len(captured) == 1
    assert captured[0].split_by_nlp == tmp_path / "subtitle-work" / "log" / "split_by_nlp.txt"


def test_synthesize_parser_accepts_expected_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["synthesize", "demo.mp4", "--output", "out.mp4"])

    assert args.command == "synthesize"
    assert args.input == "demo.mp4"
    assert args.output == "out.mp4"
    assert callable(args.func)


def test_download_parser_accepts_force_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["download", "https://example.com/demo", "--force"])

    assert args.command == "download"
    assert args.url == "https://example.com/demo"
    assert args.force is True


def test_synthesize_returns_success_for_existing_inputs(tmp_path: Path) -> None:
    input_file = tmp_path / "demo.mp4"
    work_dir = tmp_path / "work-dir"
    work_dir.mkdir()
    input_file.write_text("video", encoding="utf-8")
    (work_dir / "src.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    (work_dir / "trans.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n", encoding="utf-8")

    synthesize.merge_subtitles_to_video = lambda *_args, **_kwargs: None

    result = synthesize.run(
        Namespace(input=str(input_file), subtitle=str(tmp_path / "ignored.srt"), output=None),
        {"work_dir": str(work_dir)},
    )

    assert result == EXIT.SUCCESS


def test_check_file_exists_skips_when_dynamic_path_exists(tmp_path: Path) -> None:
    target_file = tmp_path / "output.txt"
    target_file.write_text("done", encoding="utf-8")
    calls = []

    @check_file_exists(lambda payload: payload["path"])
    def wrapped(payload):
        calls.append(payload)

    wrapped({"path": target_file})
    assert calls == []


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
    assert captured_opts["convertsubtitles"] == "srt"
    assert captured_opts["postprocessors"] == [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}]
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


def test_download_allows_missing_subtitle_and_thumbnail(monkeypatch, tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    (workspace_path / "info.json").write_text("{}", encoding="utf-8")

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download):
            origin_root = Path(self.opts["outtmpl"]["default"]).parent
            (origin_root / "video.mp4").write_text("video", encoding="utf-8")
            assert download is True

    monkeypatch.setattr("my_video.core.yt_dlp_download.yt_dlp.YoutubeDL", FakeYoutubeDL)

    result = download(DownloadRequest(url="https://example.com/demo", workspace_path=workspace_path))

    assert result.video_path == str((workspace_path / "origin" / "video.mp4").resolve())
    assert result.subtitle_path is None
    assert result.thumbnail_path is None
    status = json.loads((workspace_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert status["origin"]["video_path"] == str((workspace_path / "origin" / "video.mp4").resolve())
    assert status["origin"]["subtitle_path"] is None


def test_download_fails_when_video_missing(monkeypatch, tmp_path: Path) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    info_path = workspace_path / "info.json"
    info_path.write_text("{}", encoding="utf-8")

    class FakeYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, _url, download):
            origin_root = Path(self.opts["outtmpl"]["default"]).parent
            (origin_root / "subtitle.srt").write_text("subtitle", encoding="utf-8")
            assert download is True

    monkeypatch.setattr("my_video.core.yt_dlp_download.yt_dlp.YoutubeDL", FakeYoutubeDL)

    try:
        download(DownloadRequest(url="https://example.com/demo", workspace_path=workspace_path))
    except FileNotFoundError as exc:
        assert str(exc) == "Video file not found after download"
    else:
        raise AssertionError("expected FileNotFoundError")
    status = json.loads((workspace_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["failed_reason"] == "Video file not found after download"
    assert status["info_path"] == str(info_path.resolve())
    assert status["origin"]["subtitle_path"] == str((workspace_path / "origin" / "subtitle.srt").resolve())


def test_sanitize_workspace_name_replaces_invalid_characters() -> None:
    assert sanitize_workspace_name(' demo:/\\\\*?"<>|title . ') == "demo_title"


def test_create_workspace_rejects_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "Demo Video"
    existing.mkdir()

    try:
        create_workspace(tmp_path, "Demo Video")
    except FileExistsError as exc:
        assert str(existing) in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def test_create_workspace_returns_expected_path(tmp_path: Path) -> None:
    workspace_path = create_workspace(tmp_path, "Demo Video")
    assert workspace_path == (tmp_path / "Demo Video").resolve()


def test_resolve_workspace_returns_expected_path(tmp_path: Path) -> None:
    workspace_path = resolve_workspace(tmp_path, "Demo Video")
    assert workspace_path == (tmp_path / "Demo Video").resolve()


def test_update_status_for_workspace_writes_expected_shape(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    update_status_for_workspace(
        status_path,
        {
            "stage": "download",
            "status": "success",
            "failed_reason": None,
            "info_path": "/tmp/info.json",
            "origin": {
                "video_path": "/tmp/video.mp4",
                "subtitle_path": None,
                "thumbnail_path": "/tmp/thumb.webp",
            },
        },
        full_update=True,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["stage"] == "download"
    assert payload["status"] == "success"
    assert payload["failed_reason"] is None
    assert payload["info_path"] == "/tmp/info.json"
    assert payload["origin"] == {
        "video_path": "/tmp/video.mp4",
        "subtitle_path": None,
        "thumbnail_path": "/tmp/thumb.webp",
    }
    assert payload["last_time"]


def test_update_status_for_workspace_deep_merges_origin(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    update_status_for_workspace(
        status_path,
        {
            "stage": "download",
            "status": "running",
            "failed_reason": None,
            "info_path": "/tmp/info.json",
            "origin": {
                "video_path": "/tmp/video.mp4",
                "subtitle_path": None,
                "thumbnail_path": None,
            },
        },
        full_update=True,
    )
    update_status_for_workspace(status_path, {"origin": {"subtitle_path": "/tmp/subtitle.srt"}})

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["origin"] == {
        "video_path": "/tmp/video.mp4",
        "subtitle_path": "/tmp/subtitle.srt",
        "thumbnail_path": None,
    }


def test_read_json_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_json(tmp_path / "missing.json") is None


def test_update_status_for_workspace_accepts_running_status(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    update_status_for_workspace(
        status_path,
        {
            "stage": "download",
            "status": "running",
            "failed_reason": None,
            "info_path": "/tmp/info.json",
            "origin": {
                "video_path": None,
                "subtitle_path": None,
                "thumbnail_path": None,
            },
        },
        full_update=True,
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "running"
    assert payload["failed_reason"] is None


def test_download_command_returns_error_when_title_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_command, "fetch_video_info", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("Failed to fetch video title")))

    result = download_command.run(
        Namespace(url="https://example.com/demo", force=False),
        {"work_dir": str(tmp_path / "workspace")},
    )

    assert result == EXIT.RUNTIME_ERROR
    assert not (tmp_path / "workspace").exists()
