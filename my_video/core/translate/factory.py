"""翻译器工厂"""

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
        return LLMTranslator(
            thread_num=thread_num,
            target_language=TargetLanguage.SIMPLIFIED_CHINESE,
            model=model,
            custom_prompt=custom_prompt,
            is_reflect=is_reflect,
        )
