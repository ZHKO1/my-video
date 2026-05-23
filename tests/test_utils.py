import json
import importlib.util
import io
from pathlib import Path
import sys

import pandas as pd

from my_video.core.asr_backend.audio_preprocess import clean_words_dataframe, extract_words_dataframe
from my_video.core.asr_backend.hallucination import (
    Token,
    format_hallucination_report,
    scan_repeated_runs,
    scan_whisperx_hallucinations,
)
from my_video.core.utils.decorator import check_file_exists
from my_video.core.utils.models import OutputPaths, build_output_paths
from my_video.core.workspace import WorkspacePaths, build_workspace_paths


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detect_whisperx_hallucination.py"


def load_hallucination_script_module():
    spec = importlib.util.spec_from_file_location("detect_whisperx_hallucination", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"failed to load script module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_workspace_paths_matches_compat_builder(tmp_path: Path) -> None:
    workspace_input = tmp_path / ".." / tmp_path.name / "workspace"
    new_paths = build_workspace_paths(workspace_input)
    compat_paths = build_output_paths(workspace_input)

    assert isinstance(new_paths, WorkspacePaths)
    assert new_paths == compat_paths
    assert new_paths.base_dir == workspace_input.expanduser().resolve()
    assert new_paths.transcribe_dir == new_paths.base_dir / "transcribe"
    assert new_paths.whisperx_json == new_paths.transcribe_dir / "whisperx.json"
    assert new_paths.word_timestamps == new_paths.transcribe_dir / "word_timestamps.xlsx"
    assert new_paths.cleaned_word_timestamps == new_paths.transcribe_dir / "cleaned_word_timestamps.xlsx"


def test_output_paths_compatibility_aliases() -> None:
    assert OutputPaths is WorkspacePaths
    assert build_output_paths is build_workspace_paths


def test_extract_words_dataframe_reads_from_whisperx_json(tmp_path: Path) -> None:
    whisperx_json = tmp_path / "whisperx.json"
    whisperx_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker_id": "spk-1",
                        "words": [
                            {"word": "»hello«", "start": 0.1, "end": 0.4},
                            {"word": "tail"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    df = extract_words_dataframe(whisperx_json)

    assert list(df.columns) == ["text", "start", "end", "speaker_id"]
    assert df.iloc[0].to_dict() == {"text": "»hello«", "start": 0.1, "end": 0.4, "speaker_id": "spk-1"}
    assert df.iloc[1]["text"] == "tail"
    assert df.iloc[1]["speaker_id"] == "spk-1"
    assert pd.isna(df.iloc[1]["start"])
    assert pd.isna(df.iloc[1]["end"])


def test_clean_words_dataframe_filters_and_fills_timestamps() -> None:
    raw_df = pd.DataFrame(
        [
            {"text": "»hello«", "start": 0.1, "end": 0.4, "speaker_id": "spk-1"},
            {"text": "", "start": 0.4, "end": 0.5, "speaker_id": "spk-1"},
            {"text": "x" * 31, "start": 0.5, "end": 0.8, "speaker_id": "spk-1"},
            {"text": "tail", "start": None, "end": None, "speaker_id": "spk-1"},
            {"text": "half", "start": 1.0, "end": None, "speaker_id": "spk-1"},
        ]
    )

    cleaned_df = clean_words_dataframe(raw_df)

    assert cleaned_df.to_dict("records") == [
        {"text": "hello", "start": 0.1, "end": 0.4, "speaker_id": "spk-1"},
        {"text": "tail", "start": 0.8, "end": 0.8, "speaker_id": "spk-1"},
        {"text": "half", "start": 1.0, "end": 1.0, "speaker_id": "spk-1"},
    ]


def test_clean_words_dataframe_fills_timestamps_before_filtering() -> None:
    raw_df = pd.DataFrame(
        [
            {"text": "lead", "start": 0.1, "end": 0.2, "speaker_id": "spk-1"},
            {"text": "x" * 31, "start": None, "end": None, "speaker_id": "spk-1"},
            {"text": "tail", "start": None, "end": None, "speaker_id": "spk-1"},
        ]
    )

    cleaned_df = clean_words_dataframe(raw_df)

    assert cleaned_df.to_dict("records") == [
        {"text": "lead", "start": 0.1, "end": 0.2, "speaker_id": "spk-1"},
        {"text": "tail", "start": 0.2, "end": 0.2, "speaker_id": "spk-1"},
    ]


def test_check_file_exists_skips_when_dynamic_path_exists(tmp_path: Path) -> None:
    target_file = tmp_path / "output.txt"
    target_file.write_text("done", encoding="utf-8")
    calls = []

    @check_file_exists(lambda payload: payload["path"])
    def wrapped(payload):
        calls.append(payload)

    wrapped({"path": target_file})
    assert calls == []


def test_check_file_exists_single_file_missing_executes_function(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    @check_file_exists(lambda p: p)
    def generate(path: Path) -> str:
        return f"generated {path}"

    result = generate(target)
    assert result == f"generated {target}"


def test_check_file_exists_single_file_existing_skips_execution(tmp_path: Path) -> None:
    target = tmp_path / "exists.txt"
    target.write_text("data")

    @check_file_exists(lambda p: p)
    def generate(path: Path) -> str:
        return f"generated {path}"

    result = generate(target)
    assert result is None


def test_check_file_exists_multiple_files_all_missing_executes(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"

    @check_file_exists(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result == "generated"


def test_check_file_exists_multiple_files_partial_existing_executes(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("data")

    @check_file_exists(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result == "generated"


def test_check_file_exists_multiple_files_all_existing_skips_execution(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("data")
    b.write_text("data")

    @check_file_exists(lambda x, y: x, lambda x, y: y)
    def generate(path_a: Path, path_b: Path) -> str:
        return "generated"

    result = generate(a, b)
    assert result is None


def test_check_file_exists_static_string_path(tmp_path: Path) -> None:
    target = tmp_path / "static.txt"
    target.write_text("data")

    @check_file_exists(str(target))
    def generate() -> str:
        return "generated"

    result = generate()
    assert result is None


def test_check_file_exists_path_like_object(tmp_path: Path) -> None:
    target = tmp_path / "pathlike.txt"
    target.write_text("data")

    @check_file_exists(target)
    def generate() -> str:
        return "generated"

    result = generate()
    assert result is None


def test_check_file_exists_preserves_function_metadata() -> None:
    @check_file_exists(lambda x: x)
    def my_func(x: Path) -> str:
        """My docstring."""
        return "done"

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."


def test_check_file_exists_lambda_receives_positional_args(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    @check_file_exists(lambda first, second: second)
    def generate(unused: Path, output: Path) -> str:
        return f"generated {output}"

    result = generate(tmp_path / "unused.txt", target)
    assert result == f"generated {target}"


def test_check_file_exists_lambda_receives_keyword_args(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    @check_file_exists(lambda unused, output: output)
    def generate(unused: Path, output: Path) -> str:
        return f"generated {output}"

    result = generate(unused=tmp_path / "unused.txt", output=target)
    assert result == f"generated {target}"


def test_scan_repeated_runs_detects_single_word_hallucination() -> None:
    tokens = [
        Token(raw="alpha", normalized="alpha", start=0.0, end=0.1),
        Token(raw="uh", normalized="uh", start=0.1, end=0.2),
        Token(raw="uh", normalized="uh", start=0.2, end=0.3),
        Token(raw="uh", normalized="uh", start=0.3, end=0.4),
        Token(raw="uh", normalized="uh", start=0.4, end=0.5),
        Token(raw="uh", normalized="uh", start=0.5, end=0.6),
        Token(raw="omega", normalized="omega", start=0.6, end=0.7),
    ]

    hits = scan_repeated_runs(
        tokens=tokens,
        segment_index=3,
        min_word_run=5,
        min_phrase_run=4,
        max_phrase_len=4,
    )

    assert len(hits) == 1
    hit = hits[0]
    assert hit.segment_index == 3
    assert hit.phrase == "uh"
    assert hit.repeat_count == 5
    assert hit.token_count == 5
    assert hit.start == 0.1
    assert hit.end == 0.6


def test_scan_whisperx_hallucinations_prefers_longer_phrase_and_formats_report(tmp_path: Path) -> None:
    input_json = tmp_path / "whisperx.json"
    input_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
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

    result = scan_whisperx_hallucinations(input_json)
    report = format_hallucination_report(result, max_examples=1)

    assert result.has_hallucination is True
    assert result.total_hits == 1
    assert result.hits[0].phrase == "you know"
    assert "WhisperX hallucination scan" in report
    assert "flagged_segments: 1" in report
    assert "phrase='you know'" in report


def test_hallucination_script_main_json_output_contains_summary_and_hits(tmp_path: Path, monkeypatch) -> None:
    module = load_hallucination_script_module()
    input_json = tmp_path / "whisperx.json"
    input_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "text": "uh uh uh uh uh",
                        "words": [
                            {"word": "uh", "start": 1.0, "end": 1.1},
                            {"word": "uh", "start": 1.1, "end": 1.2},
                            {"word": "uh", "start": 1.2, "end": 1.3},
                            {"word": "uh", "start": 1.3, "end": 1.4},
                            {"word": "uh", "start": 1.4, "end": 1.5},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), str(input_json), "--json"])
    monkeypatch.setattr(sys, "stdout", stdout)

    exit_code = module.main()

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["input_json"] == str(input_json)
    assert payload["summary"]["total_segments"] == 1
    assert payload["summary"]["flagged_segments"] == 1
    assert payload["summary"]["total_hits"] == 1
    assert payload["hits"][0]["phrase"] == "uh"
    assert payload["hits"][0]["repeat_count"] == 5
