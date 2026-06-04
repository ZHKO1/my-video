from argparse import Namespace
import builtins
import json
from pathlib import Path
import sys

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import subtitle
from my_video.core.asr.asr_data import ASRData, ASRDataSeg, SubtitleLine


class InteractiveStdin:
    def isatty(self) -> bool:
        return True


class NonInteractiveStdin:
    def isatty(self) -> bool:
        return False


def _write_whisperx(path: Path, words: list[dict]) -> None:
    path.write_text(json.dumps({"word_segments": words}), encoding="utf-8")


def _install_common_patches(monkeypatch, captured: dict[str, object]) -> None:
    from my_video.core.analysis import summary as summary_module
    from my_video.core.optimize.optimize import SubtitleOptimizer
    from my_video.core.optimize.punctuation import PunctuationOptimizer
    from my_video.core.split.split import SubtitleSplitter
    from my_video.core.translate.factory import TranslatorFactory

    monkeypatch.setattr(summary_module, "get_summary", lambda *args, **kwargs: {"summary": "ok", "terms": []})
    monkeypatch.setattr(PunctuationOptimizer, "optimize", lambda self, data: data)
    monkeypatch.setattr(
        SubtitleOptimizer,
        "optimize_subtitle",
        lambda self, data, reference_data=None: data,
    )
    monkeypatch.setattr(
        SubtitleSplitter,
        "split_subtitle",
        lambda self, groups: _capture_split_groups(captured, groups),
    )
    monkeypatch.setattr(subtitle, "write_subtitle_lines_to_srt", lambda *args, **kwargs: None)

    class DummyTranslator:
        def translate_subtitle(self, subtitle_lines):
            captured["subtitle_lines"] = subtitle_lines
            return subtitle_lines

    monkeypatch.setattr(
        TranslatorFactory,
        "create_translator",
        lambda **kwargs: DummyTranslator(),
    )


def _capture_split_groups(captured: dict[str, object], groups):
    captured["split_groups"] = groups
    return [
        SubtitleLine(
            group_index=group.index,
            line_index=0,
            text=group.text,
            start_time=group.segments[0].start_time,
            end_time=group.segments[-1].end_time,
        )
        for group in groups
    ]


def test_subtitle_reuses_existing_summary_json(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)

    _write_whisperx(
        transcribe_dir / "whisperx.json",
        [
            {"word": "hello", "start": 0.0, "end": 0.1},
            {"word": "world.", "start": 0.1, "end": 0.2},
        ],
    )
    (subtitle_dir / "summary.json").write_text(
        json.dumps({"summary": "cached", "terms": []}),
        encoding="utf-8",
    )

    from my_video.core.analysis import summary as summary_module

    captured: dict[str, object] = {}
    _install_common_patches(monkeypatch, captured)
    monkeypatch.setattr(
        summary_module,
        "get_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call LLM")),
    )
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    answers = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.SUCCESS
    assert not (subtitle_dir / "optimized.txt").exists()
    assert [line.text for line in captured["subtitle_lines"]] == ["hello world."]


def test_subtitle_passes_origin_subtitle_data_to_optimizer(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    origin_dir = workspace / "origin"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)
    origin_dir.mkdir(parents=True)

    _write_whisperx(
        transcribe_dir / "whisperx.json",
        [
            {"word": "Let's", "start": 0.0, "end": 0.1},
            {"word": "try", "start": 0.1, "end": 0.2},
            {"word": "plugging", "start": 0.2, "end": 0.3},
            {"word": "in", "start": 0.3, "end": 0.4},
            {"word": "the", "start": 0.4, "end": 0.5},
            {"word": "USB", "start": 0.5, "end": 0.6},
            {"word": "charger.", "start": 0.6, "end": 0.8},
        ],
    )
    subtitle_path = origin_dir / "subtitle.srt"
    subtitle_path.write_text(
        "1\n00:00:00,000 --> 00:00:00,800\nLet's try plus in the USB.\n",
        encoding="utf-8",
    )
    (workspace / "status.json").write_text(
        json.dumps({"origin": {"subtitle_path": str(subtitle_path)}}),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    _install_common_patches(monkeypatch, captured)
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    from my_video.core.optimize.optimize import SubtitleOptimizer

    seen: dict[str, ASRData | None] = {}

    def fake_optimize(self, data, reference_data=None):
        seen["reference_data"] = reference_data
        return data

    monkeypatch.setattr(SubtitleOptimizer, "optimize_subtitle", fake_optimize)
    answers = iter(["", "Y"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.SUCCESS
    assert isinstance(seen["reference_data"], ASRData)
    assert seen["reference_data"].to_txt() == "Let's try plus in the USB."
    assert (subtitle_dir / "optimized.txt").exists()


def test_subtitle_rebuilds_sentence_groups_after_optimization(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    origin_dir = workspace / "origin"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)
    origin_dir.mkdir(parents=True)

    _write_whisperx(
        transcribe_dir / "whisperx.json",
        [
            {"word": "hello", "start": 0.0, "end": 0.1},
            {"word": "world", "start": 0.1, "end": 0.2},
        ],
    )
    from my_video.core.asr.asr_data import ASRSentenceData, SentenceGroup
    from my_video.core.optimize.optimize import SubtitleOptimizer

    def fake_optimize(self, data, reference_data=None):
        return ASRSentenceData(
            [
                SentenceGroup(
                    index=0,
                    segments=[
                        ASRDataSeg("Hello.", 0, 100),
                        ASRDataSeg("World.", 100, 200),
                    ],
                    text="Hello. World.",
                )
            ]
        )

    captured: dict[str, object] = {}
    _install_common_patches(monkeypatch, captured)
    monkeypatch.setattr(SubtitleOptimizer, "optimize_subtitle", fake_optimize)
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    answers = iter(["", "Y"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.SUCCESS
    split_groups = captured["split_groups"]
    assert [group.text for group in split_groups] == ["Hello.", "World."]


def test_subtitle_saves_punctuation_output_when_enabled(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)

    _write_whisperx(
        transcribe_dir / "whisperx.json",
        [
            {"word": "hello", "start": 0.0, "end": 0.1},
            {"word": "world", "start": 0.1, "end": 0.2},
        ],
    )

    from my_video.core.optimize.punctuation import PunctuationOptimizer

    captured: dict[str, object] = {}
    _install_common_patches(monkeypatch, captured)
    monkeypatch.setattr(
        PunctuationOptimizer,
        "optimize",
        lambda self, data: ASRData([ASRDataSeg("hello,", 0, 100), ASRDataSeg("world.", 100, 200)]),
    )
    monkeypatch.setattr(sys, "stdin", InteractiveStdin())
    answers = iter(["Y", ""])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.SUCCESS
    assert (subtitle_dir / "punctuation.txt").read_text(encoding="utf-8") == (
        "【hello/hello,】 【world/world.】"
    )
    assert not (subtitle_dir / "optimized.txt").exists()


def test_subtitle_errors_when_prompt_needed_but_stdin_is_not_interactive(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)

    _write_whisperx(
        transcribe_dir / "whisperx.json",
        [
            {"word": "hello", "start": 0.0, "end": 0.1},
            {"word": "world.", "start": 0.1, "end": 0.2},
        ],
    )

    captured: dict[str, object] = {}
    _install_common_patches(monkeypatch, captured)
    monkeypatch.setattr(sys, "stdin", NonInteractiveStdin())

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.RUNTIME_ERROR
