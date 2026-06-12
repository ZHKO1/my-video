"""subtitle command."""

from __future__ import annotations

import sys
import traceback
from argparse import Namespace
from pathlib import Path

from my_video.cli import EXIT, output
from my_video.cli.config import get_toml_value
from my_video.core.asr.text_diff import normalize_whitespace, render_inline_diff
from my_video.core.subtitle_io import write_subtitle_lines_to_srt
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

    from my_video.core.asr.asr_data import ASRData
    from my_video.core.optimize.optimize import SubtitleOptimizer
    from my_video.core.split.split import SubtitleSplitter
    from my_video.core.translate.factory import TranslatorFactory
    from my_video.core.utils.text_utils import is_mainly_cjk

    try:
        asr_data = ASRData.from_whisperx_json(str(paths.whisperx_json))
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
        sentence_data = new_sentence_data.to_asr_data().to_sentence_data()

        splitter = SubtitleSplitter(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
            max_word_count=max_word_count,
        )
        subtitle_lines = splitter.split_subtitle(sentence_data.sentences)

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

        write_subtitle_lines_to_srt(
            translated_lines, paths.src_srt, use_translation=False
        )
        write_subtitle_lines_to_srt(
            translated_lines, paths.trans_srt, use_translation=True
        )

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

    from my_video.core.asr.asr_data import ASRData

    return ASRData.from_subtitle_file(str(subtitle_path))


def _write_stage_diff(
    *, save_path: Path, original_text: str, updated_text: str
) -> None:
    diff_text = render_inline_diff(
        normalize_whitespace(original_text),
        normalize_whitespace(updated_text),
        mode="strict",
        display="reference",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(diff_text, encoding="utf-8")


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
