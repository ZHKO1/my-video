"""
翻译模块

提供多种翻译服务: OpenAI LLM、Google、Bing、DeepLX
"""

from my_video.core.entities import SubtitleProcessData
from my_video.core.translate.base import BaseTranslator
from my_video.core.translate.factory import TranslatorFactory
from my_video.core.translate.llm_translator import LLMTranslator
from my_video.core.translate.types import TargetLanguage, TranslatorType

__all__ = [
    "BaseTranslator",
    "SubtitleProcessData",
    "TranslatorFactory",
    "TranslatorType",
    "TargetLanguage",
    "LLMTranslator",
]
