from pathlib import Path
import re
from typing import List

from my_video.core.entities import SubtitleLayoutEnum

class ASRDataSeg:
    def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = ""):
        self.text = text
        self.translated_text = translated_text
        self.start_time = start_time
        self.end_time = end_time
    def __str__(self) -> str:
        return f"ASRDataSeg({self.text}, {self.start_time}, {self.end_time})"

class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        self.segments = filtered_segments

    @staticmethod
    def _format_srt_timestamp(milliseconds: int) -> str:
        hours, remainder = divmod(max(milliseconds, 0), 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, millis = divmod(remainder, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def to_txt(
        self,
        save_path=None,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.TRANSLATE_ON_TOP,
    ) -> str:
        """Convert to plain text subtitle format (without timestamps)"""
        result = []
        for seg in self.segments:
            original = seg.text
            translated = seg.translated_text

            if layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                text = f"{original}\n{translated}" if translated else original
            elif layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP:
                text = f"{translated}\n{original}" if translated else original
            elif layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                text = original
            else:  # ONLY_TRANSLATE
                text = translated if translated else original
            result.append(text)
        text = "\n".join(result)
        if save_path:
            # save_path = handle_long_path(save_path)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(result))
        return text

    def to_srt(
        self,
        save_path=None,
        layout: SubtitleLayoutEnum = SubtitleLayoutEnum.TRANSLATE_ON_TOP,
    ) -> str:
        """Convert to SRT subtitle format."""
        blocks = []
        for index, seg in enumerate(self.segments, 1):
            original = seg.text.strip()
            translated = seg.translated_text.strip()

            if layout == SubtitleLayoutEnum.ORIGINAL_ON_TOP:
                text = f"{original}\n{translated}" if translated else original
            elif layout == SubtitleLayoutEnum.TRANSLATE_ON_TOP:
                text = f"{translated}\n{original}" if translated else original
            elif layout == SubtitleLayoutEnum.ONLY_ORIGINAL:
                text = original
            else:  # ONLY_TRANSLATE
                if not translated:
                    raise ValueError(f"Missing translated_text for subtitle segment {index}")
                text = translated

            blocks.append(
                "\n".join(
                    [
                        str(index),
                        f"{self._format_srt_timestamp(seg.start_time)} --> {self._format_srt_timestamp(seg.end_time)}",
                        text,
                    ]
                )
            )

        srt_text = "\n\n".join(blocks) + ("\n" if blocks else "")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
        return srt_text

    @staticmethod
    def from_subtitle_file(file_path: str)->"ASRData":
        """Load ASRData from subtitle file.

        Args:
            file_path: Subtitle file path (supports .srt, .vtt, .ass, .json)

        Returns:
            Parsed ASRData instance

        Raises:
            FileNotFoundError: File does not exist
            ValueError: Unsupported file format
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path_obj.read_text(encoding="gbk")
        
        suffix = file_path_obj.suffix.lower()

        if suffix == ".srt":
            return ASRData.from_srt(content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
    @staticmethod
    def from_srt(srt_str: str) -> "ASRData":
        segments = []
        srt_time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
        )
        blocks = re.split(r"\n\s*\n", srt_str.strip())
        for block in blocks:
            lines = block.splitlines()
            if len(lines) < 3:
                continue

            match = srt_time_pattern.match(lines[1])
            if not match:
                continue

            time_parts = list(map(int, match.groups()))
            start_time = sum(
                [
                    time_parts[0] * 3600000,
                    time_parts[1] * 60000,
                    time_parts[2] * 1000,
                    time_parts[3],
                ]
            )
            end_time = sum(
                [
                    time_parts[4] * 3600000,
                    time_parts[5] * 60000,
                    time_parts[6] * 1000,
                    time_parts[7],
                ]
            )

            text_lines = lines[2:]
            if len(text_lines) == 1:
                segments.append(ASRDataSeg(text_lines[0], start_time, end_time))
            else:
                # Multi-line subtitle: preserve line breaks with \n
                segments.append(ASRDataSeg("\n".join(text_lines), start_time, end_time))

        return ASRData(segments)
    
    def remove_punctuation(self) -> "ASRData":
        """Remove trailing Chinese punctuation (comma, period) from segments."""
        punctuation = r"[，。]"
        for seg in self.segments:
            seg.text = re.sub(f"{punctuation}+$", "", seg.text.strip())
            seg.translated_text = re.sub(
                f"{punctuation}+$", "", seg.translated_text.strip()
            )
        return self
