import json
import re
from dataclasses import dataclass
from pathlib import Path

from my_video.core.utils.helper import write_srt
from my_video.core.utils.text_utils import count_words

SENTENCE_END_PATTERN = re.compile(r"[?!.？！。]+$")
SRT_TRAILING_H_PATTERN = re.compile(r"(?:\\h)+$")


class SubtitleSegment:
    def __init__(self, text: str, start_time: int, end_time: int):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


@dataclass
class SubtitleSentence:
    index: int
    segments: list["SubtitleSegment"]
    text: str
    optimized_text: str = ""
    optimize_log: str = ""


def build_sentence_list(segments: list["SubtitleSegment"]) -> list[SubtitleSentence]:
    groups: list[SubtitleSentence] = []
    current_group: list[SubtitleSegment] = []

    for seg in segments:
        current_group.append(seg)
        if SENTENCE_END_PATTERN.search(seg.text.strip()):
            groups.append(
                SubtitleSentence(
                    index=len(groups),
                    segments=current_group,
                    text=" ".join(segment.text.strip() for segment in current_group),
                )
            )
            current_group = []

    if current_group:
        groups.append(
            SubtitleSentence(
                index=len(groups),
                segments=current_group,
                text=" ".join(segment.text.strip() for segment in current_group),
            )
        )

    return groups


@dataclass
class SubtitleLine:
    sentence_index: int
    sentence_splited_line_index: int
    line_index: int
    text: str
    translate_text: str = ""
    start_time: int = 0
    end_time: int = 0


class SubtitleLines:
    def __init__(self, lines: list[SubtitleLine]):
        self.lines = list(lines)

    def to_txt(self, save_path: str | Path | None = None) -> str:
        output_lines: list[str] = []
        current_sentence_index: int | None = None
        current_group: list[SubtitleLine] = []

        def flush_group() -> None:
            if not current_group:
                return
            sentence_index = current_group[0].sentence_index
            if len(current_group) == 1:
                output_lines.append(f"{sentence_index}. {current_group[0].text}")
            else:
                output_lines.append(f"{sentence_index}. ")
                for line in current_group:
                    output_lines.append(f"【{count_words(line.text)}】{line.text}")

        for line in self.lines:
            if current_sentence_index is None:
                current_sentence_index = line.sentence_index
            if line.sentence_index != current_sentence_index:
                flush_group()
                current_group = []
                current_sentence_index = line.sentence_index
            current_group.append(line)

        flush_group()
        content = "\n".join(output_lines)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return content

    def to_srt(self, path: str | Path, is_translation: bool) -> None:
        write_srt(
            path,
            (
                (
                    index,
                    line.start_time,
                    line.end_time,
                    line.translate_text if is_translation else line.text,
                )
                for index, line in enumerate(self.lines, start=1)
            ),
        )


class SubtitleSegments:
    def __init__(self, segments: list[SubtitleSegment]):
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: (x.start_time, x.end_time))
        self.segments = filtered_segments

    def to_txt(
        self,
        save_path=None,
    ) -> str:
        """Convert to plain text subtitle format (without timestamps)"""
        result = []
        for seg in self.segments:
            original = seg.text
            result.append(original)
        text = " ".join(result)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
        return text

    def to_sentence_data(self) -> "SubtitleSentences":
        return SubtitleSentences(build_sentence_list(self.segments))

    @staticmethod
    def from_whisperx_json(file_path: str) -> "SubtitleSegments":
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        try:
            content = file_path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path_obj.read_text(encoding="gbk")

        data = json.loads(content)
        word_segments = data.get("word_segments", [])

        segments = []
        for entry in word_segments:
            word = entry.get("word", "").strip()
            if not word:
                continue
            start = entry.get("start")
            end = entry.get("end")
            if start is None or end is None:
                continue
            segments.append(
                SubtitleSegment(
                    text=word,
                    start_time=int(start * 1000),
                    end_time=int(end * 1000),
                )
            )

        for seg in segments:
            if any(c.isspace() for c in seg.text):
                raise ValueError(
                    f"Invalid word segment(contain space at middle) at {seg.start_time}ms-{seg.end_time}ms: '{seg.text}'"
                )
            # if not any(c.isalnum() for c in seg.text):
            #     raise ValueError(
            #         f"Invalid word segment(pure punctuation) at {seg.start_time}ms-{seg.end_time}ms: '{seg.text}'"
            #     )

        return SubtitleSegments(segments)

    @staticmethod
    def from_subtitle_file(file_path: str) -> "SubtitleSegments":
        """Load SubtitleSegments from subtitle file.

        Args:
            file_path: Subtitle file path (supports .srt, .vtt, .ass, .json)

        Returns:
            Parsed SubtitleSegments instance

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
            return SubtitleSegments.from_srt(content)
        # 需要实际例子参考
        # elif suffix == ".vtt":
        #     if "<c>" in content:
        #         return SubtitleSegments.from_youtube_vtt(content)
        #     return SubtitleSegments.from_vtt(content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_srt(srt_str: str) -> "SubtitleSegments":
        """Create SubtitleSegments from SRT format string.

        Uses language detection to distinguish between bilingual subtitles
        (original + translation) and multiline single-language subtitles.

        Args:
            srt_str: SRT format subtitle string

        Returns:
            Parsed SubtitleSegments instance
        """
        segments = []
        srt_time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})\s-->\s(\d{2}):(\d{2}):(\d{1,2})[.,](\d{3})"
        )
        blocks = re.split(r"\n\s*\n", srt_str.strip())

        # Process all blocks based on detected mode
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

            text_lines = [
                SRT_TRAILING_H_PATTERN.sub("", text_line).strip()
                for text_line in lines[2:]
            ]
            if len(text_lines) == 1:
                segments.append(SubtitleSegment(text_lines[0], start_time, end_time))
            else:
                segments.append(
                    SubtitleSegment(" ".join(text_lines), start_time, end_time)
                )

        return SubtitleSegments(segments)

    @staticmethod
    def from_vtt(vtt_str: str) -> "SubtitleSegments":
        """Create SubtitleSegments from VTT format string.

        Args:
            vtt_str: VTT format subtitle string

        Returns:
            SubtitleSegments instance
        """
        segments = []
        # Split by blank lines, skip the WEBVTT header block
        blocks = vtt_str.strip().split("\n\n")
        # Find first block after header (skip WEBVTT line and any NOTE/STYLE blocks)
        content = []
        header_done = False
        for block in blocks:
            stripped = block.strip()
            if not header_done:
                if (
                    stripped.startswith("WEBVTT")
                    or stripped.startswith("NOTE")
                    or stripped.startswith("STYLE")
                ):
                    continue
                header_done = True
            if stripped:
                content.append(stripped)

        # Support both HH:MM:SS.mmm and MM:SS.mmm (VTT allows omitting hours)
        timestamp_pattern = re.compile(
            r"(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d{2}):)?(\d{2}):(\d{2})\.(\d{3})"
        )

        for block in content:
            lines = block.split("\n")
            if not lines:
                continue

            # Find the timestamp line (could be first line or second if cue ID present)
            timestamp_line = None
            text_start = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    timestamp_line = line
                    text_start = i + 1
                    break

            if not timestamp_line:
                continue
            match = timestamp_pattern.match(timestamp_line.strip())
            if not match:
                continue

            groups = match.groups()
            time_parts = [int(g) if g is not None else 0 for g in groups]
            start_time = (
                time_parts[0] * 3600000
                + time_parts[1] * 60000
                + time_parts[2] * 1000
                + time_parts[3]
            )
            end_time = (
                time_parts[4] * 3600000
                + time_parts[5] * 60000
                + time_parts[6] * 1000
                + time_parts[7]
            )

            text_line = " ".join(lines[text_start:])
            # Remove VTT inline tags: timestamps, <c>, <b>, <i>, <u>, <ruby>, etc.
            cleaned_text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text_line)
            cleaned_text = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            if cleaned_text and cleaned_text != " ":
                segments.append(SubtitleSegment(cleaned_text, start_time, end_time))

        return SubtitleSegments(segments)

    @staticmethod
    def from_youtube_vtt(vtt_str: str) -> "SubtitleSegments":
        """Create SubtitleSegments from YouTube VTT format with word-level timestamps.

        Args:
            vtt_str: YouTube VTT format subtitle string (contains <c> tags)

        Returns:
            Parsed SubtitleSegments with word-level segments
        """

        def parse_timestamp(ts: str) -> int:
            """Convert timestamp string to milliseconds"""
            h, m, s = ts.split(":")
            return int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)

        def split_timestamped_text(text: str) -> list[SubtitleSegment]:
            """Extract word segments from timestamped text"""
            pattern = re.compile(r"<(\d{2}:\d{2}:\d{2}\.\d{3})>([^<]*)")
            matches = list(pattern.finditer(text))
            word_segments = []

            for i in range(len(matches) - 1):
                current_match = matches[i]
                next_match = matches[i + 1]

                start_time = parse_timestamp(current_match.group(1))
                end_time = parse_timestamp(next_match.group(1))
                word = current_match.group(2).strip()

                if word:
                    word_segments.append(SubtitleSegment(word, start_time, end_time))

            return word_segments

        segments = []
        blocks = re.split(r"\n\n+", vtt_str.strip())

        timestamp_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})"
        )
        for block in blocks:
            lines = block.strip().split("\n")
            if not lines:
                continue

            match = timestamp_pattern.match(lines[0])
            if not match:
                continue

            text = "\n".join(lines)

            timestamp_row = re.search(r"\n(.*?<c>.*?</c>.*)", block)
            if timestamp_row:
                text = re.sub(r"<c>|</c>", "", timestamp_row.group(1))
                block_start_time_string = (
                    f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
                )
                block_end_time_string = (
                    f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
                )
                text = f"<{block_start_time_string}>{text}<{block_end_time_string}>"

                word_segments = split_timestamped_text(text)
                segments.extend(word_segments)

        return SubtitleSegments(segments)


class SubtitleSentences:
    def __init__(self, sentences: list[SubtitleSentence]):
        self.sentences = list(sentences)

    def to_txt(self, save_path: str | Path | None = None) -> str:
        lines: list[str] = []
        for sentence in self.sentences:
            lines.append(
                f"{sentence.index}. {sentence.optimize_log or sentence.optimized_text or sentence.text}"
            )

        text = "\n".join(lines)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_asr_data(self) -> "SubtitleSegments":
        segments: list[SubtitleSegment] = []
        for sentence in self.sentences:
            segments.extend(
                SubtitleSegment(
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                )
                for segment in sentence.segments
            )
        return SubtitleSegments(segments)
