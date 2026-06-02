"""subtitle command."""

from argparse import Namespace
from pathlib import Path
import traceback

from my_video.cli import exit_codes as EXIT
from my_video.cli import output
from my_video.cli.config import get_toml_value
from my_video.core.subtitle_io import write_subtitle_lines_to_srt
from my_video.core.workspace import build_workspace_paths
from my_video.core.utils.helper import write_json


def run(args: Namespace, config: dict) -> int:
    workspace_path = Path(args.workspace_path).expanduser()
    if not workspace_path.is_dir():
        output.error(f"Workspace not found: {workspace_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    if not paths.whisperx_json.exists():
        output.error(f"whisperx.json not found: {str(paths.whisperx_json)}")
        return EXIT.FILE_NOT_FOUND

    from my_video.core.analysis.summary import get_summary
    from my_video.core.asr.asr_data import ASRData
    from my_video.core.optimize.optimize import SubtitleOptimizer
    from my_video.core.optimize.punctuation import PunctuationOptimizer
    from my_video.core.split.split import SubtitleSplitter
    from my_video.core.translate.factory import TranslatorFactory
    from my_video.core.utils.text_utils import is_mainly_cjk

    asr_data = ASRData.from_whisperx_json(str(paths.whisperx_json))

    try:
        thread_num = get_toml_value(config, "subtitle.thread_num", 4)
        batch_size = get_toml_value(config, "subtitle.batch_size", 20)
        llm_model = get_toml_value(config, "llm.model", "deepseek-v4-pro")
        need_reflect = get_toml_value(config, "subtitle.need_reflect", True)
        max_sentence_word_count_english = get_toml_value(
            config, "subtitle.max_sentence_word_count_english", 50
        )
        max_sentence_word_count_cjk = get_toml_value(
            config, "subtitle.max_sentence_word_count_cjk", 50
        )
        max_cjk = get_toml_value(config, "subtitle.max_word_count_cjk", 16)
        max_english = get_toml_value(config, "subtitle.max_word_count_english", 18)

        is_cjk = is_mainly_cjk("".join(seg.text for seg in asr_data.segments))
        max_sentence_word_count = (
            max_sentence_word_count_cjk if is_cjk else max_sentence_word_count_english
        )
        max_word_count = max_cjk if is_cjk else max_english

        punctuation_optimizer = PunctuationOptimizer(
            thread_num=thread_num,
            model=llm_model,
            max_sentence_word_count=max_sentence_word_count,
        )
        asr_data = punctuation_optimizer.optimize(asr_data)

        sentence_data = asr_data.to_sentence_data()

        optimizer = SubtitleOptimizer(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
        )
        optimized_sentence_data = optimizer.optimize_subtitle(sentence_data)
        optimized_sentence_data.to_txt(paths.optimized_txt)

        splitter = SubtitleSplitter(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
            max_word_count=max_word_count,
        )
        subtitle_lines = splitter.split_subtitle(optimized_sentence_data.sentences)

        if not paths.summary_json.exists():
            summary = get_summary(paths.optimized_txt, model=llm_model)
            write_json(paths.summary_json, summary)

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
