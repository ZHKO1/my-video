from argparse import Namespace
import json
from pathlib import Path

import pandas as pd

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import transcribe
from my_video.cli.main import build_parser
from my_video.core.workspace import build_workspace_paths


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
    monkeypatch.setattr(
        transcribe,
        "convert_video_to_audio",
        lambda video_file, output_path: captured.setdefault("convert_video_to_audio", (video_file, output_path)),
    )

    def fake_whisperx_audio(audio_file, whisperx_json, whisper_language, model_name, model_dir):
        Path(whisperx_json).parent.mkdir(parents=True, exist_ok=True)
        Path(whisperx_json).write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker_id": "spk-1",
                            "words": [
                                {"word": "»hello«", "start": 0.1, "end": 0.4},
                                {"word": "x" * 31, "start": 0.4, "end": 0.8},
                                {"word": "tail"},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(transcribe, "whisperx_audio", fake_whisperx_audio)
    monkeypatch.setattr(
        transcribe,
        "save_dataframe",
        lambda df, output_path: captured.setdefault("saved_frames", []).append((df.copy(), output_path)),
    )

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    assert result == EXIT.SUCCESS
    assert captured["workspace_path"] == workspace_path
    assert captured["convert_video_to_audio"] == (
        str(input_file),
        str(workspace_path.resolve() / "transcribe" / "raw.mp3"),
    )
    assert json.loads((workspace_path / "transcribe" / "whisperx.json").read_text(encoding="utf-8")) == {
        "segments": [
            {
                "speaker_id": "spk-1",
                "words": [
                    {"word": "»hello«", "start": 0.1, "end": 0.4},
                    {"word": "x" * 31, "start": 0.4, "end": 0.8},
                    {"word": "tail"},
                ],
            }
        ]
    }

    saved_frames = captured["saved_frames"]
    assert len(saved_frames) == 2

    raw_df, raw_path = saved_frames[0]
    assert raw_path == workspace_path.resolve() / "transcribe" / "word_timestamps.xlsx"
    assert raw_df.iloc[0].to_dict() == {"text": "»hello«", "start": 0.1, "end": 0.4, "speaker_id": "spk-1"}
    assert raw_df.iloc[1].to_dict() == {"text": "x" * 31, "start": 0.4, "end": 0.8, "speaker_id": "spk-1"}
    assert raw_df.iloc[2]["text"] == "tail"
    assert raw_df.iloc[2]["speaker_id"] == "spk-1"
    assert pd.isna(raw_df.iloc[2]["start"])
    assert pd.isna(raw_df.iloc[2]["end"])

    cleaned_df, cleaned_path = saved_frames[1]
    assert cleaned_path == workspace_path.resolve() / "transcribe" / "cleaned_word_timestamps.xlsx"
    assert cleaned_df.to_dict("records") == [
        {"text": "hello", "start": 0.1, "end": 0.4, "speaker_id": "spk-1"},
        {"text": "tail", "start": 0.8, "end": 0.8, "speaker_id": "spk-1"},
    ]


def test_transcribe_returns_error_when_whisperx_hallucination_detected(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    input_file = tmp_path / "origin.mp4"
    input_file.write_text("x")
    (workspace_path / "status.json").write_text(
        json.dumps({"origin": {"video_path": str(input_file)}}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        transcribe,
        "convert_video_to_audio",
        lambda video_file, output_path: captured.setdefault("convert_video_to_audio", (video_file, output_path)),
    )

    def fake_whisperx_audio(audio_file, whisperx_json, whisper_language, model_name, model_dir):
        Path(whisperx_json).parent.mkdir(parents=True, exist_ok=True)
        Path(whisperx_json).write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker_id": "spk-1",
                            "words": [
                                {"word": "you", "start": 0.0, "end": 0.1},
                                {"word": "know", "start": 0.1, "end": 0.2},
                                {"word": "you", "start": 0.2, "end": 0.3},
                                {"word": "know", "start": 0.3, "end": 0.4},
                                {"word": "you", "start": 0.4, "end": 0.5},
                                {"word": "know", "start": 0.5, "end": 0.6},
                                {"word": "you", "start": 0.6, "end": 0.7},
                                {"word": "know", "start": 0.7, "end": 0.8},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(transcribe, "whisperx_audio", fake_whisperx_audio)
    monkeypatch.setattr(
        transcribe,
        "save_dataframe",
        lambda df, output_path: captured.setdefault("saved_frames", []).append((df.copy(), output_path)),
    )

    result = transcribe.run(Namespace(workspace_path=str(workspace_path)), {})

    stderr = capsys.readouterr().err
    assert result == EXIT.RUNTIME_ERROR
    assert "WhisperX hallucination scan" in stderr
    assert "flagged_segments: 1" in stderr
    assert "phrase='you know'" in stderr
    assert "WhisperX hallucination detected" in stderr
    assert "saved_frames" not in captured


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
