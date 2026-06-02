"""词级标点优化模块。"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import json_repair

from my_video.cli import output
from my_video.core.asr.asr_data import (
    ASRData,
    ASRDataSeg,
    SentenceGroup,
    batch_sentence_groups,
    build_sentence_groups,
    is_long_sentence_group,
)
from my_video.core.llm import call_llm
from my_video.core.prompts import get_prompt
from my_video.core.utils.helper import split_token_parts, split_tokens

SEGMENT_WORD_THRESHOLD=500
MAX_STEPS = 3
ALLOWED_PUNCTUATION = ",.?"


class PunctuationBatch(NamedTuple):
    groups: List[SentenceGroup]

    @property
    def payload(self) -> Dict[str, str]:
        return {str(group.index): group.text for group in self.groups}


class PunctuationOptimizer:
    def __init__(self, thread_num: int, model: str, max_sentence_word_count: int):
        self.thread_num = thread_num
        self.model = model
        self.max_sentence_word_count = max_sentence_word_count
        self.is_running = True
        self.executor: Optional[ThreadPoolExecutor] = None
        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def optimize(self, subtitle_data: Union[str, ASRData]) -> ASRData:
        try:
            if isinstance(subtitle_data, str):
                asr_data = ASRData.from_whisperx_json(subtitle_data)
            else:
                asr_data = subtitle_data

            sentence_groups = build_sentence_groups(asr_data.segments)
            candidate_groups = [
                group
                for group in sentence_groups
                if is_long_sentence_group(group, self.max_sentence_word_count)
            ]

            if not candidate_groups:
                return ASRData(self._copy_segments(asr_data.segments))

            batches = self._batch_sentence_groups(candidate_groups)
            optimized_groups = self._parallel_optimize(batches)
            return self._write_back_groups(sentence_groups, optimized_groups)
        except Exception as e:
            output.error(f"Punctuation optimization failed: {str(e)}")
            raise RuntimeError(f"Punctuation optimization failed: {str(e)}")

    def _batch_sentence_groups(
        self, groups: List[SentenceGroup], threshold: int = SEGMENT_WORD_THRESHOLD
    ) -> List[PunctuationBatch]:
        return [
            PunctuationBatch(groups=batch)
            for batch in batch_sentence_groups(groups, threshold)
        ]

    def _parallel_optimize(self, batches: List[PunctuationBatch]) -> Dict[int, str]:
        if not self.executor:
            raise ValueError("Thread pool not initialized")

        futures = []
        optimized_groups: Dict[int, str] = {}

        for batch in batches:
            future = self.executor.submit(self._optimize_batch, batch)
            futures.append((future, batch))

        for future, batch in futures:
            if not self.is_running:
                break

            try:
                optimized_groups.update(future.result())
            except Exception as e:
                output.error(f"Punctuation batch failed: {str(e)}")
                optimized_groups.update({group.index: group.text for group in batch.groups})

        return optimized_groups

    def _optimize_batch(self, batch: PunctuationBatch) -> Dict[int, str]:
        payload = batch.payload
        start_idx = batch.groups[0].index
        end_idx = batch.groups[-1].index
        output.info(f"[+]Optimizing punctuation: {start_idx} - {end_idx}")

        result = self._agent_loop(payload)
        return {int(key): value for key, value in result.items()}

    def _agent_loop(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        user_prompt = (
            "Add punctuation to the following long sentence groups. "
            "Return only modified keys as pure JSON.\n"
            f"<input_subtitle>{subtitle_chunk}</input_subtitle>"
        )

        messages = [
            {"role": "system", "content": get_prompt("optimize/punctuation")},
            {"role": "user", "content": user_prompt},
        ]

        last_result: Optional[Dict[str, str]] = None

        for step in range(MAX_STEPS):
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.1,
            )
            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("LLM returned empty result")

            parsed_result = json_repair.loads(result_text)
            if not isinstance(parsed_result, dict):
                raise ValueError(
                    f"LLM returned invalid type, expected dict, got {type(parsed_result)}"
                )

            result_dict = {str(key): str(value) for key, value in parsed_result.items()}
            last_result = result_dict

            is_valid, error_message = self._validate_optimization_result(
                original_chunk=subtitle_chunk,
                optimized_chunk=result_dict,
            )
            if is_valid:
                merged_result = dict(subtitle_chunk)
                merged_result.update(result_dict)
                return merged_result

            output.warn(
                f"Punctuation validation failed, feedback loop ({step + 1}): {error_message}"
            )
            messages.append({"role": "assistant", "content": result_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {error_message}\n"
                        "Fix the errors and output ONLY a valid JSON object containing modified keys."
                    ),
                }
            )

        output.warn(f"Max attempts reached({MAX_STEPS}), returning last valid fallback")
        merged_result = dict(subtitle_chunk)
        return merged_result

    def _validate_optimization_result(
        self, original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Tuple[bool, str]:
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        if not actual_keys.issubset(expected_keys):
            extra = sorted(actual_keys - expected_keys)
            return (
                False,
                f"Extra keys: {extra}. Only return a subset of {sorted(expected_keys)}.",
            )

        for key, optimized_text in optimized_chunk.items():
            original_text = original_chunk[key]
            # original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            # optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()
            # similarity = difflib.SequenceMatcher(
            #     None, original_cleaned, optimized_cleaned
            # ).ratio()
            # similarity_threshold = 0.85 if count_words(original_text) <= 10 else 0.92

            # if similarity < similarity_threshold:
            #     return (
            #         False,
            #         f"Key '{key}' changed too much: similarity {similarity:.1%} < {similarity_threshold:.0%}.",
            #     )

            is_valid_rewrite, rewrite_error = self._is_valid_punctuation_rewrite(
                original_text.strip(), optimized_text.strip()
            )
            if not is_valid_rewrite:
                return (
                    False,
                    f"Key '{key}' contains disallowed edits: {rewrite_error}",
                )

        return True, ""

    @classmethod
    def _is_valid_punctuation_rewrite(
        cls, original_text: str, optimized_text: str
    ) -> Tuple[bool, str]:
        original_tokens = split_tokens(original_text)
        optimized_tokens = split_tokens(optimized_text)

        if len(original_tokens) != len(optimized_tokens):
            return (
                False,
                f"token count mismatch: original={len(original_tokens)}, optimized={len(optimized_tokens)}",
            )

        for index, (original_token, optimized_token) in enumerate(
            zip(original_tokens, optimized_tokens),
            start=1,
        ):
            original_base = split_token_parts(original_token, ALLOWED_PUNCTUATION).base
            optimized_base = split_token_parts(optimized_token, ALLOWED_PUNCTUATION).base

            if original_base.lower() != optimized_base.lower():
                return (
                    False,
                    f"token {index} mismatch: '{original_token}' -> '{optimized_token}'",
                )

        return True, ""

    def _write_back_groups(
        self,
        sentence_groups: List[SentenceGroup],
        optimized_groups: Dict[int, str],
    ) -> ASRData:
        new_segments: List[ASRDataSeg] = []
        for group in sentence_groups:
            optimized_text = optimized_groups.get(group.index)
            if optimized_text is None or optimized_text == group.text:
                new_segments.extend(self._copy_segments(group.segments))
                continue

            rewritten_segments = self._copy_segments(group.segments)
            self._apply_text_to_segments(rewritten_segments, optimized_text)
            new_segments.extend(rewritten_segments)

        return ASRData(new_segments)

    @classmethod
    def _apply_text_to_segments(cls, segments: List[ASRDataSeg], optimized_text: str) -> None:
        original_text = " ".join(seg.text.strip() for seg in segments)
        is_valid_rewrite, rewrite_error = cls._is_valid_punctuation_rewrite(
            original_text, optimized_text
        )
        if not is_valid_rewrite:
            raise ValueError(
                f"Optimized text cannot be aligned back to original segments: {rewrite_error}"
            )

        optimized_tokens = split_tokens(optimized_text)
        if len(segments) != len(optimized_tokens):
            raise ValueError("Optimized token count does not match original segments")

        for index, (seg, optimized_token) in enumerate(zip(segments, optimized_tokens)):
            seg.text = f"{optimized_token} " if index < len(optimized_tokens) - 1 else optimized_token

    @staticmethod
    def _copy_segments(segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(seg.text, seg.start_time, seg.end_time)
            for seg in segments
        ]

    def stop(self) -> None:
        if not self.is_running:
            return

        self.is_running = False
        if self.executor:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            finally:
                self.executor = None
