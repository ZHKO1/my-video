from argparse import Namespace
import json
from pathlib import Path

from my_video.cli import exit_codes as EXIT
from my_video.cli.commands import subtitle


def test_subtitle_reuses_existing_summary_json(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    transcribe_dir = workspace / "transcribe"
    subtitle_dir = workspace / "subtitle"
    transcribe_dir.mkdir(parents=True)
    subtitle_dir.mkdir(parents=True)

    (transcribe_dir / "whisperx.json").write_text(
        json.dumps(
            {
                "word_segments": [
                    {"word": "hello", "start": 0.0, "end": 0.1},
                    {"word": "world.", "start": 0.1, "end": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    (subtitle_dir / "summary.json").write_text(
        json.dumps({"summary": "cached", "terms": []}),
        encoding="utf-8",
    )

    from my_video.core.analysis import summary as summary_module
    from my_video.core.optimize.optimize import SubtitleOptimizer
    from my_video.core.optimize.punctuation import PunctuationOptimizer
    from my_video.core.translate.factory import TranslatorFactory

    monkeypatch.setattr(summary_module, "get_summary", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call LLM")))
    monkeypatch.setattr(PunctuationOptimizer, "optimize", lambda self, data: data)
    monkeypatch.setattr(SubtitleOptimizer, "optimize_subtitle", lambda self, data: data)

    captured: dict[str, object] = {}

    class DummyTranslator:
        def translate_subtitle(self, sentence_data):
            captured["sentence_data"] = sentence_data

    monkeypatch.setattr(
        TranslatorFactory,
        "create_translator",
        lambda **kwargs: DummyTranslator(),
    )

    result = subtitle.run(Namespace(workspace_path=str(workspace)), {"subtitle": {}, "llm": {}})

    assert result == EXIT.SUCCESS
    assert (subtitle_dir / "optimized.txt").read_text(encoding="utf-8") == "0. hello world."
    assert [group.text for group in captured["sentence_data"].sentences] == ["hello world."]
