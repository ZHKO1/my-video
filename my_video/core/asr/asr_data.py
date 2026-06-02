import json
from dataclasses import dataclass, field
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
    optimize_logs: list[str] = field(default_factory=list)

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
            lines.append(f"{sentence.index}. {sentence.optimized_text or sentence.text}")
            lines.extend(f"  {log}" for log in sentence.optimize_logs)

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
                    "optimize_logs": list(sentence.optimize_logs),
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
                optimize_logs=[str(item) for item in sentence.get("optimize_logs", [])],
                segments=[
                    ASRDataSeg.from_dict(segment)
                    for segment in sentence.get("segments", [])
                ],
            )
            for sentence in payload.get("sentences", [])
        ]
        return cls(sentences)


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
            if not any(c.isalnum() for c in seg.text):
                raise ValueError(
                    f"Invalid word segment(pure punctuation) at {seg.start_time}ms-{seg.end_time}ms: '{seg.text}'"
                )

        return ASRData(segments)
