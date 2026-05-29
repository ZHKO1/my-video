from dataclasses import dataclass
from enum import Enum

class SubtitleLayoutEnum(Enum):
    """字幕布局"""

    TRANSLATE_ON_TOP = "译文在上"
    ORIGINAL_ON_TOP = "原文在上"
    ONLY_ORIGINAL = "仅原文"
    ONLY_TRANSLATE = "仅译文"


@dataclass
class SubtitleProcessData:
    """字幕处理数据（翻译/优化通用）"""

    index: int
    original_text: str
    translated_text: str = ""
    optimized_text: str = ""
