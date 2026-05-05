from argparse import Namespace
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import transcribe
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
