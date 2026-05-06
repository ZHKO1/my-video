from argparse import Namespace
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.core.utils.decorator import check_file_exists
from my_video.cli.commands import subtitle, transcribe
from my_video.cli.main import build_parser


def test_transcribe_parser_does_not_define_output_option() -> None:
    parser = build_parser()
    with_output_format = parser.parse_args(["transcribe", "input.mp4", "--format", "srt"])
    assert with_output_format.format == "srt"

    try:
        parser.parse_args(["transcribe", "input.mp4", "--output", "somewhere"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected --output to be rejected")


def test_transcribe_uses_work_dir_from_config(monkeypatch, tmp_path: Path) -> None:
    input_file = tmp_path / "demo.mp4"
    input_file.write_text("x")
    expected = str(tmp_path / "work-dir")
    captured: list[str] = []

    def fake_build_output_paths(base_dir: str):
        captured.append(base_dir)

        class DummyPaths:
            output_dir = tmp_path / "generated"

        return DummyPaths()

    monkeypatch.setattr(transcribe, "build_output_paths", fake_build_output_paths)
    monkeypatch.setattr(
        transcribe,
        "convert_video_to_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
        raising=False,
    )

    result = transcribe.run(Namespace(input=str(input_file), verbose=False, quiet=True), {"work_dir": expected})

    assert result == EXIT.RUNTIME_ERROR
    assert captured == [expected]


def test_transcribe_falls_back_to_current_directory_without_work_dir(monkeypatch, tmp_path: Path) -> None:
    input_file = tmp_path / "demo.mp4"
    input_file.write_text("x")
    captured: list[str] = []

    def fake_build_output_paths(base_dir: str):
        captured.append(base_dir)

        class DummyPaths:
            output_dir = tmp_path / "generated"

        return DummyPaths()

    monkeypatch.setattr(transcribe, "build_output_paths", fake_build_output_paths)
    monkeypatch.setattr(
        transcribe,
        "convert_video_to_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
        raising=False,
    )

    result = transcribe.run(Namespace(input=str(input_file), verbose=False, quiet=True), {})

    assert result == EXIT.RUNTIME_ERROR
    assert captured == ["."]


def test_subtitle_parser_accepts_input_and_common_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["subtitle", "demo.srt", "-q"])

    assert args.command == "subtitle"
    assert args.input == "demo.srt"
    assert args.quiet is True
    assert callable(args.func)


def test_subtitle_returns_success_for_existing_input(tmp_path: Path) -> None:
    input_file = tmp_path / "demo.srt"
    input_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    def fake_split_by_spacy(paths):
        paths.split_by_nlp.write_text("ok", encoding="utf-8")

    subtitle.split_by_spacy = fake_split_by_spacy
    result = subtitle.run(
        Namespace(input=str(input_file), verbose=False, quiet=True),
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

    result = subtitle.run(
        Namespace(input=str(input_file), verbose=False, quiet=True),
        {"work_dir": str(tmp_path / "subtitle-work")},
    )

    assert result == EXIT.SUCCESS
    assert len(captured) == 1
    assert captured[0].split_by_nlp == tmp_path / "subtitle-work" / "log" / "split_by_nlp.txt"


def test_check_file_exists_skips_when_dynamic_path_exists(tmp_path: Path) -> None:
    target_file = tmp_path / "output.txt"
    target_file.write_text("done", encoding="utf-8")
    calls = []

    @check_file_exists(lambda payload: payload["path"])
    def wrapped(payload):
        calls.append(payload)

    wrapped({"path": target_file})
    assert calls == []
