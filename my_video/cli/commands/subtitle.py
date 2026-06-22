"""subtitle command."""

from __future__ import annotations

import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path

from my_video.cli import EXIT, output
from my_video.cli.config import get_toml_value
from my_video.core.utils.helper import read_json
from my_video.core.workspace import build_workspace_paths


def run(args: Namespace, config: dict) -> int:
    workspace_path = Path(args.workspace_path).expanduser()
    if not workspace_path.is_dir():
        output.error(f"Workspace not found: {workspace_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    if not paths.whisperx_json.exists():
        output.error(f"whisperx.json not found: {paths.whisperx_json!s}")
        return EXIT.FILE_NOT_FOUND

    from my_video.core.asr.asr_data import SubtitleSegments
    from my_video.core.optimize.optimize import SubtitleOptimizer
    from my_video.core.split.split import SubtitleSplitter
    from my_video.core.translate.factory import TranslatorFactory
    from my_video.core.utils.text_utils import is_mainly_cjk

    try:
        asr_data = SubtitleSegments.from_whisperx_json(str(paths.whisperx_json))
        reference_data = _load_origin_subtitle_data(paths)

        thread_num = get_toml_value(config, "subtitle.thread_num", 4)
        batch_size = get_toml_value(config, "subtitle.batch_size", 20)
        llm_model = get_toml_value(config, "llm.model", "deepseek-v4-pro")
        need_reflect = get_toml_value(config, "subtitle.need_reflect", True)
        max_cjk = get_toml_value(config, "subtitle.max_word_count_cjk", 16)
        max_english = get_toml_value(config, "subtitle.max_word_count_english", 18)

        is_cjk = is_mainly_cjk("".join(seg.text for seg in asr_data.segments))
        max_word_count = max_cjk if is_cjk else max_english

        sentence_data = asr_data.to_sentence_data()
        sentence_data.to_txt(paths.origin_txt)

        optimizer = SubtitleOptimizer(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
        )
        new_sentence_data = optimizer.optimize_subtitle(
            sentence_data,
            reference_data=reference_data,
        )

        sentence_data.to_txt(paths.optimized_txt)
        _write_tmp_optimized_json(_repo_root(), sentence_data)
        # 优化的文本后可能有标点符号的改动，所以需要重新分句号
        sentence_data = new_sentence_data.to_asr_data().to_sentence_data()

        splitter = SubtitleSplitter(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
            max_word_count=max_word_count,
        )
        subtitle_lines = splitter.split_subtitle(sentence_data.sentences)
        subtitle_lines.to_txt(paths.split_txt)

        # if not paths.summary_json.exists():
        #     summary_input_path = paths.subtitle_dir / ".summary_input.txt"
        #     sentence_data.to_txt(summary_input_path)
        #     summary = get_summary(summary_input_path, model=llm_model)
        #     write_json(paths.summary_json, summary)

        translator = TranslatorFactory.create_translator(
            thread_num=thread_num,
            model=llm_model,
            custom_prompt="",
            is_reflect=need_reflect,
        )
        translated_lines = translator.translate_subtitle(subtitle_lines)

        translated_lines.to_srt(paths.src_srt, is_translation=False)
        translated_lines.to_srt(paths.trans_srt, is_translation=True)

        output.success(f"Subtitle files saved to {paths.src_srt} and {paths.trans_srt}")
        return EXIT.SUCCESS

    except Exception as e:
        output.error(e)
        output.error(traceback.format_exc())
        return EXIT.RUNTIME_ERROR


def _load_origin_subtitle_data(paths):
    status = read_json(paths.status_path) or {}
    origin = status.get("origin")
    subtitle_path = origin.get("subtitle_path") if isinstance(origin, dict) else None
    if not subtitle_path:
        return None

    from my_video.core.asr.asr_data import SubtitleSegments

    return SubtitleSegments.from_subtitle_file(str(subtitle_path))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_tmp_optimized_json(root_path: Path, sentence_data) -> None:
    export_path = root_path / "tmp.json"
    payload = [
        {
            "txt": sentence.text,
            "optimized_text": sentence.optimized_text,
        }
        for sentence in sentence_data.sentences
    ]
    export_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prompt_choice(
    prompt: str, valid_choices: set[str], default: str | None = None
) -> str:
    if not sys.stdin.isatty():
        raise RuntimeError(f"stdin is not interactive, cannot prompt: {prompt}")

    while True:
        print(prompt, file=sys.stderr, end=" ", flush=True)
        answer = input().strip()
        if not answer and default is not None:
            return default
        if answer in valid_choices:
            return answer
        output.warn(f"Invalid input: {answer or '<empty>'}")


def _prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    default_choice = None
    if default is True:
        default_choice = "Y"
    elif default is False:
        default_choice = "N"
    return (
        _prompt_choice(
            prompt,
            {"Y", "y", "N", "n"},
            default=default_choice,
        ).lower()
        == "y"
    )
