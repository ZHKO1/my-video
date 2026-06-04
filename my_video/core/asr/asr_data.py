import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, List, TypeVar

from my_video.core.utils.helper import read_json_source, write_json_source
from my_video.core.utils.text_utils import count_words

SENTENCE_END_PATTERN = re.compile(r"[?!.？！。]+$")
T = TypeVar("T")


@dataclass
class SentenceGroup:
    index: int
    segments: List["ASRDataSeg"]
    text: str
    optimized_text: str = ""
    optimize_log: str = ""

    def check_segment_gaps(self, max_gap_ms: int = 2000) -> tuple[bool, list[int]]:
        gap_positions: list[int] = []
        for position, (previous_seg, current_seg) in enumerate(
            zip(self.segments, self.segments[1:]),
            start=1,
        ):
            if current_seg.start_time - previous_seg.end_time > max_gap_ms:
                gap_positions.append(position)
        return bool(gap_positions), gap_positions

    def format_with_gap_markers(self, gap_positions: list[int]) -> str:
        if not self.segments:
            return ""

        gap_position_set = set(gap_positions)
        parts: list[str] = []
        for position, seg in enumerate(self.segments, start=1):
            parts.append(seg.text.strip())
            if position < len(self.segments):
                parts.append("】【" if position in gap_position_set else " ")
        return "".join(parts)


def batch_items_by_word_count(
    items: List[T],
    get_text: Callable[[T], str],
    threshold: int,
) -> List[List[T]]:
    batches: List[List[T]] = []
    current_batch: List[T] = []
    current_count = 0

    for item in items:
        count = count_words(get_text(item))
        if current_batch and current_count + count > threshold:
            batches.append(current_batch)
            current_batch = []
            current_count = 0

        current_batch.append(item)
        current_count += count

    if current_batch:
        batches.append(current_batch)

    return batches


def build_sentence_groups(segments: List["ASRDataSeg"]) -> List[SentenceGroup]:
    groups: List[SentenceGroup] = []
    current_group: List[ASRDataSeg] = []

    for seg in segments:
        current_group.append(seg)
        if SENTENCE_END_PATTERN.search(seg.text.strip()):
            groups.append(
                SentenceGroup(
                    index=len(groups),
                    segments=current_group,
                    text=" ".join(segment.text.strip() for segment in current_group),
                )
            )
            current_group = []

    if current_group:
        groups.append(
            SentenceGroup(
                index=len(groups),
                segments=current_group,
                text=" ".join(segment.text.strip() for segment in current_group),
            )
        )

    return groups


def batch_sentence_groups(
    groups: List[SentenceGroup],
    threshold: int = 500,
) -> List[List[SentenceGroup]]:
    return batch_items_by_word_count(
        groups,
        get_text=lambda group: group.text,
        threshold=threshold,
    )


def is_long_sentence_group(group: SentenceGroup, max_sentence_word_count: int) -> bool:
    return count_words(group.text) > max_sentence_word_count


class ASRDataSeg:
    def __init__(self, text: str, start_time: int, end_time: int):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self) -> str:
        return f"ASRDataSeg({self.text}, {self.start_time}, {self.end_time})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASRDataSeg":
        return cls(
            text=str(data["text"]),
            start_time=int(data["start_time"]),
            end_time=int(data["end_time"]),
        )


@dataclass
class SubtitleLine:
    group_index: int
    line_index: int
    text: str
    translate_text: str = ""
    start_time: int = 0
    end_time: int = 0

    @property
    def line_id(self) -> str:
        return f"{self.group_index}:{self.line_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_index": self.group_index,
            "line_index": self.line_index,
            "text": self.text,
            "translate_text": self.translate_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleLine":
        return cls(
            group_index=int(data["group_index"]),
            line_index=int(data["line_index"]),
            text=str(data["text"]),
            translate_text=str(data.get("translate_text", "")),
            start_time=int(data["start_time"]),
            end_time=int(data["end_time"]),
        )


class ASRSentenceData:
    def __init__(self, sentences: list[SentenceGroup]):
        self.sentences = list(sentences)

    def to_txt(self, save_path: str | Path | None = None) -> str:
        lines: list[str] = []
        for sentence in self.sentences:
            lines.append(f"{sentence.index}. {sentence.optimize_log or sentence.optimized_text or sentence.text}")

        text = "\n".join(lines)
        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    def to_json(self, save_path: str | Path | None = None) -> str:
        payload = {
            "sentences": [
                {
                    "index": sentence.index,
                    "text": sentence.text,
                    "optimized_text": sentence.optimized_text,
                    "optimize_log": sentence.optimize_log,
                    "segments": [segment.to_dict() for segment in sentence.segments],
                }
                for sentence in self.sentences
            ]
        }
        return write_json_source(payload, save_path)

    @classmethod
    def from_json(cls, source: str | Path | dict[str, Any]) -> "ASRSentenceData":
        payload = read_json_source(source)
        sentences = [
            SentenceGroup(
                index=int(sentence["index"]),
                text=str(sentence["text"]),
                optimized_text=str(sentence.get("optimized_text", "")),
                optimize_log=str(sentence.get("optimize_log", "")),
                segments=[
                    ASRDataSeg.from_dict(segment)
                    for segment in sentence.get("segments", [])
                ],
            )
            for sentence in payload.get("sentences", [])
        ]
        return cls(sentences)

    def to_asr_data(self) -> "ASRData":
        segments: list[ASRDataSeg] = []
        for sentence in self.sentences:
            segments.extend(
                ASRDataSeg(
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                )
                for segment in sentence.segments
            )
        return ASRData(segments)


class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        filtered_segments = [seg for seg in segments if seg.text and seg.text.strip()]
        filtered_segments.sort(key=lambda x: x.start_time)
        self.segments = filtered_segments

    def to_json(self, save_path: str | Path | None = None) -> str:
        payload = {
            "segments": [segment.to_dict() for segment in self.segments],
        }
        return write_json_source(payload, save_path)

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

    @classmethod
    def from_json(cls, source: str | Path | dict[str, Any]) -> "ASRData":
        payload = read_json_source(source)
        return cls(
            [ASRDataSeg.from_dict(segment) for segment in payload.get("segments", [])]
        )

    def to_sentence_data(self) -> ASRSentenceData:
        return ASRSentenceData(build_sentence_groups(self.segments))

    @staticmethod
    def from_whisperx_json(file_path: str) -> "ASRData":
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
                ASRDataSeg(
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

        return ASRData(segments)

    @staticmethod
    def from_subtitle_file(file_path: str) -> "ASRData":
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
        elif suffix == ".vtt":
            if "<c>" in content:
                return ASRData.from_youtube_vtt(content)
            return ASRData.from_vtt(content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def from_srt(srt_str: str) -> "ASRData":
        """Create ASRData from SRT format string.

        Uses language detection to distinguish between bilingual subtitles
        (original + translation) and multiline single-language subtitles.

        Args:
            srt_str: SRT format subtitle string

        Returns:
            Parsed ASRData instance
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

            text_lines = lines[2:]
            if len(text_lines) == 1:
                segments.append(ASRDataSeg(text_lines[0], start_time, end_time))
            else:
                # Multi-line subtitle: preserve line breaks with \n
                segments.append(ASRDataSeg(" ".join(text_lines), start_time, end_time))

        return ASRData(segments)

    @staticmethod
    def from_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from VTT format string.

        Args:
            vtt_str: VTT format subtitle string

        Returns:
            ASRData instance
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
                if stripped.startswith("WEBVTT") or stripped.startswith("NOTE") or stripped.startswith("STYLE"):
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
                time_parts[0] * 3600000 + time_parts[1] * 60000 +
                time_parts[2] * 1000 + time_parts[3]
            )
            end_time = (
                time_parts[4] * 3600000 + time_parts[5] * 60000 +
                time_parts[6] * 1000 + time_parts[7]
            )

            text_line = " ".join(lines[text_start:])
            # Remove VTT inline tags: timestamps, <c>, <b>, <i>, <u>, <ruby>, etc.
            cleaned_text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text_line)
            cleaned_text = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned_text)
            cleaned_text = cleaned_text.strip()

            if cleaned_text and cleaned_text != " ":
                segments.append(ASRDataSeg(cleaned_text, start_time, end_time))

        return ASRData(segments)

    @staticmethod
    def from_youtube_vtt(vtt_str: str) -> "ASRData":
        """Create ASRData from YouTube VTT format with word-level timestamps.

        Args:
            vtt_str: YouTube VTT format subtitle string (contains <c> tags)

        Returns:
            Parsed ASRData with word-level segments
        """

        def parse_timestamp(ts: str) -> int:
            """Convert timestamp string to milliseconds"""
            h, m, s = ts.split(":")
            return int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)

        def split_timestamped_text(text: str) -> List[ASRDataSeg]:
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
                    word_segments.append(ASRDataSeg(word, start_time, end_time))

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

        return ASRData(segments)
