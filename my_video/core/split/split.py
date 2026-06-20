import atexit
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import json_repair
from rapidfuzz.distance import Levenshtein

from my_video.cli import output
from my_video.core.asr.asr_data import SubtitleLine, SubtitleSegment, SubtitleSentence
from my_video.core.llm import call_llm, get_response_id
from my_video.core.prompts import get_prompt
from my_video.core.utils.text_utils import count_words, is_mainly_cjk

MAX_STEPS = 3
CONTENT_SIMILARITY_THRESHOLD = 0.95
MATCH_SIMILARITY_THRESHOLD = 0.8
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class SplitRequest:
    group_index: int
    text: str


class SubtitleSplitter:
    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        model: str,
        custom_prompt: str,
        max_word_count: int,
    ) -> None:
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.model = model
        self.custom_prompt = custom_prompt
        self.max_word_count = max_word_count
        self.is_running = True
        self.executor: ThreadPoolExecutor | None = None
        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def split_subtitle(
        self, sentence_groups: list[SubtitleSentence]
    ) -> list[SubtitleLine]:
        requests = [
            SplitRequest(group_index=group.index, text=group.text)
            for group in sentence_groups
            if not self._is_within_limit(group.text)
        ]
        split_results = self._process_requests(self._batch_requests(requests))

        subtitle_lines: list[SubtitleLine] = []
        for group in sentence_groups:
            parts = split_results.get(group.index, [group.text])
            subtitle_lines.extend(self._build_subtitle_lines(group, parts))
        return subtitle_lines

    def _batch_requests(self, requests: list[SplitRequest]) -> list[dict[str, str]]:
        return [
            {
                str(request.group_index): request.text
                for request in requests[i : i + self.batch_num]
            }
            for i in range(0, len(requests), self.batch_num)
        ]

    def _process_requests(self, batches: list[dict[str, str]]) -> dict[int, list[str]]:
        if not self.executor:
            raise ValueError("Thread pool not initialized")
        if not batches:
            return {}

        futures = [
            self.executor.submit(self._process_batch, batch) for batch in batches
        ]
        split_results: dict[int, list[str]] = {}

        for future in as_completed(futures):
            if not self.is_running:
                break
            try:
                split_results.update(future.result())
            except Exception:
                output.error(traceback.format_exc())
                raise

        return split_results

    def _process_batch(self, batch: dict[str, str]) -> dict[int, list[str]]:
        start_idx = next(iter(batch))
        end_idx = next(reversed(batch))
        output.info(f"[+]Spliting subtitles: {start_idx} - {end_idx}")

        try:
            return self._agent_loop(batch)
        except Exception as exc:
            output.error(f"Split failed: {exc}")
            return {int(key): [text] for key, text in batch.items()}

    def _agent_loop(self, batch: dict[str, str]) -> dict[int, list[str]]:
        user_prompt = (
            "Split the following subtitle groups with <br> separators. "
            "Keep the original language and do not rewrite.\n"
            f"<input_subtitle>{json.dumps(batch, ensure_ascii=False)}</input_subtitle>"
        )
        if self.custom_prompt:
            user_prompt += (
                f"\nReference content:\n<reference>{self.custom_prompt}</reference>"
            )

        messages = [
            {
                "role": "system",
                "content": get_prompt(
                    "split/structured", max_word_count=self.max_word_count
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        last_result: dict[int, list[str]] | None = None
        for step in range(MAX_STEPS):
            response = call_llm(messages=messages, model=self.model, temperature=0.2)
            response_id = get_response_id(response)
            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("LLM returned empty result")

            try:
                parsed_result = json_repair.loads(result_text)
            except Exception as exc:
                is_valid = False
                error_message = f"JSON parse error: {exc}"
            else:
                is_valid, error_message = self._validate_json(parsed_result, batch)
                if is_valid:
                    parsed_parts = self._parse_split_result(parsed_result)
                    last_result = parsed_parts
                    is_valid, error_message = self._validate_split_result(
                        batch, parsed_parts
                    )
                    if is_valid:
                        return parsed_parts

            output.warn(
                f"分割验证失败[{response_id}]，开始反馈循环 (第{step + 1}次尝试): {error_message}"
            )
            messages.append({"role": "assistant", "content": result_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {error_message}\n"
                        "Please fix the errors and output ONLY a valid JSON dictionary."
                    ),
                }
            )

        output.warn(f"Max attempts reached({MAX_STEPS})，returning last result")
        return (
            last_result
            if last_result
            else {int(key): [text] for key, text in batch.items()}
        )

    def _validate_json(
        self, parsed_result: Any, batch: dict[str, str]
    ) -> tuple[bool, str]:
        if not isinstance(parsed_result, dict):
            return (
                False,
                f"JSON structure error: expected dict, got {type(parsed_result)}",
            )
        if set(parsed_result.keys()) != set(batch.keys()):
            return (
                False,
                "JSON structure error: response keys do not match request keys.",
            )

        for key, value in parsed_result.items():
            if not isinstance(value, str):
                return (
                    False,
                    f"JSON structure error: value for key {key} must be a string.",
                )
            parts = self._split_response_text(value)
            if not parts:
                return (
                    False,
                    f"JSON structure error: value for key {key} must contain at least one non-empty part.",
                )

        return True, ""

    def _parse_split_result(
        self, parsed_result: dict[str, str]
    ) -> dict[int, list[str]]:
        return {
            int(key): self._split_response_text(value)
            for key, value in parsed_result.items()
        }

    def _validate_split_result(
        self,
        original_batch: dict[str, str],
        split_chunk: dict[int, list[str]],
    ) -> tuple[bool, str]:
        for key, original_text in original_batch.items():
            parts = split_chunk.get(int(key))
            if not parts:
                return (
                    False,
                    f"Content validation error: missing split parts for key {key}.",
                )

            merged = self._join_parts(parts)
            content_feedback = self._build_content_feedback(
                original_text,
                merged,
                label=f"key {key}",
            )
            if content_feedback:
                return False, content_feedback

            violations = [
                f"part {index + 1} has {count_words(part)} words > {self.max_word_count}"
                for index, part in enumerate(parts)
                if count_words(part) > self.max_word_count
            ]
            if violations:
                return (
                    False,
                    f"Length violations for key {key}: {'; '.join(violations)}",
                )

        return True, ""

    def _build_subtitle_lines(
        self, group: SubtitleSentence, parts: list[str]
    ) -> list[SubtitleLine]:
        try:
            segment_ranges = self._match_parts_to_segments(group.segments, parts)
        except Exception as exc:
            output.warn(
                f"Failed to align split result for group {group.index}, fallback to single line: {exc}"
            )
            segment_ranges = [(0, len(group.segments) - 1)]
            parts = [group.text]

        lines: list[SubtitleLine] = []
        for line_index, ((start_idx, end_idx), text) in enumerate(
            zip(segment_ranges, parts, strict=True)
        ):
            segs = group.segments[start_idx : end_idx + 1]
            lines.append(
                SubtitleLine(
                    group_index=group.index,
                    line_index=line_index,
                    text=text,
                    start_time=segs[0].start_time,
                    end_time=segs[-1].end_time,
                )
            )
        return lines

    def _match_parts_to_segments(
        self,
        segments: list[SubtitleSegment],
        parts: list[str],
    ) -> list[tuple[int, int]]:
        if not segments:
            raise ValueError("Sentence group has no segments")
        if len(parts) == 1:
            return [(0, len(segments) - 1)]

        matches: list[tuple[int, int]] = []
        cursor = 0

        for part_index, part in enumerate(parts):
            if part_index == len(parts) - 1:
                final_text = self._segments_to_text(segments[cursor:], part)
                final_score = self._text_similarity(part, final_text)
                if final_score < MATCH_SIMILARITY_THRESHOLD:
                    raise ValueError(
                        f"cannot align final part with segments (score={final_score:.1%})"
                    )
                matches.append((cursor, len(segments) - 1))
                break

            remaining_parts = len(parts) - part_index - 1
            candidate_stop = len(segments) - remaining_parts
            best_score = -1.0
            best_end: int | None = None

            for end_idx in range(cursor, candidate_stop):
                candidate_text = self._segments_to_text(
                    segments[cursor : end_idx + 1], part
                )
                score = self._text_similarity(part, candidate_text)
                if score > best_score:
                    best_score = score
                    best_end = end_idx
                if score == 1.0:
                    break

            if best_end is None or best_score < MATCH_SIMILARITY_THRESHOLD:
                raise ValueError(
                    f"cannot align part {part_index} with segments (score={best_score:.1%})"
                )

            matches.append((cursor, best_end))
            cursor = best_end + 1

        if cursor > len(segments):
            raise ValueError("segment alignment overflow")
        return matches

    def _segments_to_text(
        self, segments: list[SubtitleSegment], reference_text: str
    ) -> str:
        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        if self._is_cjk_text(reference_text):
            return "".join(texts)
        return " ".join(texts)

    def _build_content_feedback(
        self, original: str, merged: str, *, label: str
    ) -> str | None:
        similarity = self._text_similarity(original, merged)
        if similarity >= CONTENT_SIMILARITY_THRESHOLD:
            return None
        return (
            f"Content changed too much in {label} (similarity: {similarity:.1%}). "
            "Only insert <br> between parts. Do not rewrite, add, or remove content."
        )

    @classmethod
    def _split_response_text(cls, text: str) -> list[str]:
        return [part.strip() for part in text.split("<br>") if part.strip()]

    def _join_parts(self, parts: list[str]) -> str:
        if not parts:
            return ""
        raw_text = "".join(parts)
        if self._is_cjk_text(raw_text):
            return raw_text
        return " ".join(part.strip() for part in parts if part.strip())

    @staticmethod
    def _normalize_text(text: str) -> str:
        stripped = text.strip()
        if SubtitleSplitter._is_cjk_text(stripped):
            return re.sub(r"\s+", "", stripped)
        return " ".join(stripped.split()).lower()

    @staticmethod
    def _canonicalize_text(text: str) -> str:
        return PUNCTUATION_PATTERN.sub("", SubtitleSplitter._normalize_text(text))

    def _text_similarity(self, left: str, right: str) -> float:
        normalized_left = self._canonicalize_text(left)
        normalized_right = self._canonicalize_text(right)
        if normalized_left == normalized_right:
            return 1.0
        if not normalized_left and not normalized_right:
            return 1.0
        return Levenshtein.normalized_similarity(normalized_left, normalized_right)

    @staticmethod
    def _is_cjk_text(text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and is_mainly_cjk(stripped)

    def _is_within_limit(self, text: str) -> bool:
        return count_words(text) <= self.max_word_count

    def stop(self) -> None:
        self.is_running = False
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
