"""翻译器工厂"""

from my_video.cli import output
from my_video.core.translate.base import BaseTranslator
from my_video.core.translate.llm_translator import LLMTranslator
from my_video.core.translate.types import TargetLanguage


class TranslatorFactory:
    """翻译器工厂类"""

    @staticmethod
    def create_translator(
        thread_num: int = 5,
        model: str = "gpt-4o-mini",
        custom_prompt: str = "",
        is_reflect: bool = False,
    ) -> BaseTranslator:
        """创建翻译器实例"""
        try:
            target_language = TargetLanguage.SIMPLIFIED_CHINESE

            return LLMTranslator(
                thread_num=thread_num,
                target_language=target_language,
                model=model,
                custom_prompt=custom_prompt,
                is_reflect=is_reflect,
            )
        except Exception as e:
            output.error(f"Failed to create translator: {e!s}")
            raise
