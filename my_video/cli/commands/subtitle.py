"""subtitle command — lightweight placeholder for subtitle processing."""

from argparse import Namespace
from pathlib import Path
import traceback


from my_video.cli import output
from my_video.cli import exit_codes as EXIT
from my_video.cli.config import get_toml_value
from my_video.core.entities import SubtitleLayoutEnum
from my_video.core.workspace import build_workspace_paths


def run(args: Namespace, config: dict) -> int:
    workspace_path = Path(args.workspace_path).expanduser()
    if not workspace_path.is_dir():
        output.error(f"Workspace not found: {workspace_path}")
        return EXIT.FILE_NOT_FOUND

    paths = build_workspace_paths(workspace_path)
    if paths.src_srt.exists() and paths.trans_srt.exists():
        output.success(f"{paths.src_srt} and {paths.trans_srt} already exist")
        return EXIT.SUCCESS

    if not paths.whisperx_json.exists():
        output.error(f"whisperx.json not found: {str(paths.whisperx_json)}")
        return EXIT.FILE_NOT_FOUND

    from my_video.core.asr.asr_data import ASRData
    asr_data = ASRData.from_whisperx_json(str(paths.whisperx_json))

    try:

        # 1. 完整台词文章，先按已有的标点符号来划分
        # 2. 每段500字左右，然后让LLM补上标点符号，同时按照完整一句来分割 批量处理
        # 3. 每一句给出完整的翻译 批量处理
        # 4. 每一句给出完整的翻译 批量处理
        # 5. 检查每句，如果过长，则启动分句逻辑，同时给出中文翻译怎么分割
        # 6. 收集成果，最后合并
        
        thread_num = get_toml_value(config, "subtitle.thread_num", 4)
        batch_size = get_toml_value(config, "subtitle.batch_size", 20)
        llm_model = get_toml_value(config, "llm.model", "deepseek-v4-pro")
        max_cjk = get_toml_value(config, "subtitle.max_word_count_cjk", 16)
        max_english = get_toml_value(config, "subtitle.max_word_count_english", 18)
        need_reflect = get_toml_value(config, "subtitle.need_reflect", True)

        # 1. 分割
        from my_video.core.split.split import SubtitleSplitter
        splitter = SubtitleSplitter(
            thread_num=thread_num,
            model=llm_model,
            max_word_count_cjk=max_cjk,
            max_word_count_english=max_english,
        )
        asr_data = splitter.split_subtitle(asr_data)

        # 2. 优化
        from my_video.core.optimize.optimize import SubtitleOptimizer
        optimizer = SubtitleOptimizer(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
        )
        asr_data = optimizer.optimize_subtitle(asr_data)
        asr_data.remove_punctuation()

        # 2. 翻译
        from my_video.core.translate.factory import TranslatorFactory
        translator = TranslatorFactory.create_translator(
            thread_num=thread_num,
            batch_num=batch_size,
            model=llm_model,
            custom_prompt="",
            is_reflect=need_reflect,
        )
        asr_data = translator.translate_subtitle(asr_data)
        asr_data.remove_punctuation()

        asr_data.to_srt(paths.src_srt, layout=SubtitleLayoutEnum.ONLY_ORIGINAL)
        asr_data.to_srt(paths.trans_srt, layout=SubtitleLayoutEnum.ONLY_TRANSLATE)
        output.success(f"Subtitle files saved to {paths.src_srt} and {paths.trans_srt}")
        return EXIT.SUCCESS

    except Exception as e:
        output.error(e)
        output.error(traceback.format_exc())
        return EXIT.RUNTIME_ERROR
