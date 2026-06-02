"""字幕优化模块

使用LLM优化字幕内容，支持agent loop自动验证和修正。
"""

import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import json_repair
from rapidfuzz.distance import Levenshtein

from my_video.cli import output
from my_video.core.utils.text_utils import count_words, is_mainly_cjk

from ..asr.asr_data import (
    ASRDataSeg,
    ASRSentenceData,
    SentenceGroup,
    batch_sentence_groups,
)
from ..llm import call_llm
from ..prompts import get_prompt
from ..utils.helper import comparison_bases_from_text, comparison_bases_from_tokens, split_token_parts, split_tokens

MAX_STEPS = 3


class SubtitleOptimizer:
    """字幕优化器

    使用LLM优化字幕内容，支持:
    - Agent loop自动验证和修正
    - 并发批量处理
    - 自动对齐修复
    """

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        model: str,
        custom_prompt: str,
    ):
        """初始化优化器

        Args:
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            model: LLM模型名称
            custom_prompt: 自定义优化提示词
            temperature: LLM温度参数
        """
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.model = model
        self.custom_prompt = custom_prompt

        self.is_running = True
        self.executor: Optional[ThreadPoolExecutor] = None
        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        """初始化线程池并注册清理函数"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def optimize_subtitle(self, subtitle_data: ASRSentenceData) -> ASRSentenceData:
        """优化字幕句组。"""
        try:
            sentence_groups = subtitle_data.sentences

            chunks = self._batch_sentence_groups(sentence_groups)

            # 并行优化
            optimized_dict = self._parallel_optimize(chunks)

            return self._write_back_groups(sentence_groups, optimized_dict)

        except Exception as e:
            output.error(f"Optimization failed: {str(e)}")
            raise RuntimeError(f"Optimization failed: {str(e)}")

    def _batch_sentence_groups(
        self, groups: List[SentenceGroup], threshold: int = 500
    ) -> List[Dict[str, str]]:
        batched_groups = batch_sentence_groups(groups, threshold)
        return [
            {str(group.index): group.text for group in batch}
            for batch in batched_groups
        ]

    def _parallel_optimize(self, chunks: List[Dict[str, str]]) -> Dict[str, str]:
        """并行优化All批次

        Args:
            chunks: 字幕批次列表

        Returns:
            优化后的字幕字典
        """
        if not self.executor:
            raise ValueError("Thread pool not initialized")

        futures = []
        optimized_dict: Dict[str, str] = {}

        # 提交All任务
        for chunk in chunks:
            future = self.executor.submit(self._optimize_chunk, chunk)
            futures.append((future, chunk))

        # 收集结果
        for future, chunk in futures:
            if not self.is_running:
                break

            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                output.error(f"Optimization batch failed: {str(e)}")
                optimized_dict.update(chunk)  # 失败时保留原文

        return optimized_dict

    def _optimize_chunk(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """优化单个字幕批次

        Args:
            subtitle_chunk: 字幕批次字典

        Returns:
            优化后的字幕批次
        """
        start_idx = next(iter(subtitle_chunk))
        end_idx = next(reversed(subtitle_chunk))
        output.info(f"[+]Optimizing subtitles: {start_idx} - {end_idx}")

        try:
            result = self.agent_loop(subtitle_chunk)

            return result

        except Exception as e:
            output.error(f"Optimization failed: {str(e)}")
            return subtitle_chunk

    def agent_loop(self, subtitle_chunk: Dict[str, str]) -> Dict[str, str]:
        """使用agent loop优化字幕

        LLM → 验证 → 反馈 → 重试 (最多MAX_STEPS次)

        Args:
            subtitle_chunk: 字幕批次字典

        Returns:
            优化后的字幕批次

        Raises:
            ValueError: LLM returned empty result
        """
        # 构建提示词
        user_prompt = (
            f"Correct the following subtitles. Keep the original language, do not translate:\n"
            f"<input_subtitle>{str(subtitle_chunk)}</input_subtitle>"
        )

        if self.custom_prompt:
            user_prompt += (
                f"\nReference content:\n<reference>{self.custom_prompt}</reference>"
            )

        messages = [
            {"role": "system", "content": get_prompt("optimize/subtitle")},
            {"role": "user", "content": user_prompt},
        ]

        last_result = None

        # Agent loop
        for step in range(MAX_STEPS):
            # 调用LLM
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=0.2,
            )

            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("LLM returned empty result")

            # 解析结果
            parsed_result = json_repair.loads(result_text)
            if not isinstance(parsed_result, dict):
                raise ValueError(
                    f"LLM返回结果类型Error，期望dict，实际{type(parsed_result)}"
                )

            result_dict: Dict[str, str] = parsed_result
            last_result = result_dict

            # 验证结果
            is_valid, error_message = self._validate_optimization_result(
                original_chunk=subtitle_chunk, optimized_chunk=result_dict
            )

            if is_valid:
                return result_dict

            # 验证失败，添加反馈
            output.warn(
                f"优化验证失败，开始反馈循环 (第{step + 1}次尝试): {error_message}"
            )
            messages.append({"role": "assistant", "content": result_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {error_message}\n"
                        f"Please fix the errors and output ONLY a valid JSON dictionary."
                    ),
                }
            )

        # 达到最大步数
        output.warn(f"Max attempts reached({MAX_STEPS})，returning last result")
        return last_result if last_result else subtitle_chunk

    def _validate_optimization_result(
        self, original_chunk: Dict[str, str], optimized_chunk: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证优化结果

        检查:
        1. 键是否完全匹配
        2. 改动是否过大（相似度 < 0.7）

        Args:
            original_chunk: 原始字幕批次
            optimized_chunk: 优化后字幕批次

        Returns:
            (是否有效, Error反馈)
        """
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        # 检查键匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys

            error_parts = []
            if missing:
                error_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {sorted(extra)}")

            error_msg = (
                "\n".join(error_parts) + f"\nRequired keys: {sorted(expected_keys)}\n"
                f"Please return the COMPLETE optimized dictionary with ALL {len(expected_keys)} keys."
            )
            return False, error_msg

        # 检查改动是否过大（逐条比较相似度）
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            original_bases = comparison_bases_from_text(original_text)
            optimized_bases = comparison_bases_from_text(optimized_text)
            similarity = Levenshtein.normalized_similarity(
                original_bases,
                optimized_bases,
            )
            similarity_threshold = 0.3 if count_words(original_text) <= 10 else 0.6

            # if similarity != 1.0:
            #     punctuation_changes = self._count_punctuation_changes(
            #         original_text,
            #         optimized_text,
            #     )
            #     output.warn(
            #         f"similarity{similarity} punct_changes={punctuation_changes}:\n"
            #         f" origin: {original_text} \n optimized: {optimized_text} \n"
            #     )

            # 相似度过低
            if similarity < similarity_threshold:
                excessive_changes.append(
                    f"Key '{key}': similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                    f"Original: '{original_text}' → Optimized: '{optimized_text}' "
                )

        if excessive_changes:
            error_msg = ";\n".join(excessive_changes)
            error_msg += (
                "\n\nYour optimizations changed the text too much. "
                "Keep high similarity (≥70% for normal text) by making MINIMAL changes: "
                "only fix recognition errors and improve clarity, "
                "but preserve the original wording, length and structure as much as possible."
            )
            return False, error_msg

        return True, ""

    @classmethod
    def _write_back_groups(
        cls,
        sentence_groups: List[SentenceGroup],
        optimized_dict: Dict[str, str],
    ) -> ASRSentenceData:
        new_groups: List[SentenceGroup] = []
        for group in sentence_groups:
            optimized_text = optimized_dict.get(str(group.index))
            if optimized_text is None or not optimized_text.strip():
                group.optimized_text = ""
                group.optimize_logs = []
                new_groups.append(
                    cls._clone_sentence_group(
                        group,
                        text=group.text,
                        optimized_text="",
                        optimize_logs=[],
                        segments=cls._copy_segments(group.segments),
                    )
                )
                continue

            if optimized_text.strip() == group.text.strip():
                group.optimized_text = ""
                group.optimize_logs = []
                new_groups.append(
                    cls._clone_sentence_group(
                        group,
                        text=group.text,
                        optimized_text="",
                        optimize_logs=[],
                        segments=cls._copy_segments(group.segments),
                    )
                )
                continue

            optimize_logs = cls._build_group_change_logs(
                original_segments=group.segments,
                original_text=group.text,
                optimized_text=optimized_text,
            )
            rewritten_segments = cls._rewrite_group_segments(
                group.segments,
                optimized_text,
            )
            group.optimized_text = optimized_text
            group.optimize_logs = optimize_logs
            new_groups.append(
                cls._clone_sentence_group(
                    group,
                    text=optimized_text,
                    optimized_text="",
                    optimize_logs=[],
                    segments=rewritten_segments,
                )
            )
        return ASRSentenceData(new_groups)

    @classmethod
    def _rewrite_group_segments(
        cls,
        original_segments: List[ASRDataSeg],
        optimized_text: str,
    ) -> List[ASRDataSeg]:
        original_tokens = [segment.text.strip() for segment in original_segments]
        optimized_tokens = split_tokens(optimized_text)
        opcodes = cls._build_token_opcodes(original_tokens, optimized_tokens)

        rewritten_segments: List[ASRDataSeg] = []
        for tag, i1, i2, j1, j2 in cls._merge_edit_opcodes(opcodes):
            if tag == "equal":
                for old_index, new_index in zip(range(i1, i2), range(j1, j2)):
                    rewritten_segments.append(
                        cls._copy_segment_with_text(
                            original_segments[old_index],
                            optimized_tokens[new_index],
                        )
                    )
                continue

            if tag == "delete":
                continue

            if tag == "insert":
                rewritten_segments.extend(
                    cls._build_insert_segments(
                        original_segments=original_segments,
                        optimized_tokens=optimized_tokens,
                        insert_at=i1,
                        token_start=j1,
                        token_end=j2,
                    )
                )
                continue

            if tag == "replace":
                rewritten_segments.extend(
                    cls._build_replace_segments(
                        original_segments=original_segments,
                        optimized_tokens=optimized_tokens,
                        original_start=i1,
                        original_end=i2,
                        token_start=j1,
                        token_end=j2,
                    )
                )

        return rewritten_segments

    @classmethod
    def _build_token_opcodes(
        cls,
        original_tokens: List[str],
        optimized_tokens: List[str],
    ) -> List[Tuple[str, int, int, int, int]]:
        original_bases = comparison_bases_from_tokens(original_tokens)
        optimized_bases = comparison_bases_from_tokens(optimized_tokens)
        return Levenshtein.opcodes(
            original_bases,
            optimized_bases,
        )

    @classmethod
    def _build_group_change_logs(
        cls,
        original_segments: List[ASRDataSeg],
        original_text: str,
        optimized_text: str,
    ) -> List[str]:
        original_units, optimized_units = cls._build_change_units(
            original_segments=original_segments,
            original_text=original_text,
            optimized_text=optimized_text,
        )
        if comparison_bases_from_tokens(original_units) == comparison_bases_from_tokens(optimized_units):
            return []

        opcodes = cls._build_token_opcodes(original_units, optimized_units)
        merged_opcodes = cls._merge_edit_opcodes(opcodes)
        logs: List[str] = []
        for opcode in merged_opcodes:
            if opcode[0] == "equal":
                continue

            change = cls._format_group_change_line(
                original_text=original_text,
                optimized_text=optimized_text,
                original_units=original_units,
                optimized_units=optimized_units,
                opcode=opcode,
            )
            if change:
                logs.append(change)
        return logs

    @classmethod
    def _build_change_units(
        cls,
        original_segments: List[ASRDataSeg],
        original_text: str,
        optimized_text: str,
    ) -> Tuple[List[str], List[str]]:
        original_tokens = [segment.text.strip() for segment in original_segments]
        text_is_cjk = (
            is_mainly_cjk(original_text)
            and " " not in original_text
            and " " not in optimized_text.strip()
        )
        if text_is_cjk:
            return list("".join(original_tokens)), list(optimized_text.strip())

        return original_tokens, split_tokens(optimized_text)

    @classmethod
    def _format_group_change_line(
        cls,
        original_text: str,
        optimized_text: str,
        original_units: List[str],
        optimized_units: List[str],
        opcode: Tuple[str, int, int, int, int],
    ) -> str:
        tag, i1, i2, j1, j2 = opcode
        original_fragment = cls._format_units(original_units[i1:i2], original_text)
        new_fragment = cls._format_units(optimized_units[j1:j2], optimized_text)

        before_context = cls._extract_context(original_text, original_units, i1, "before")
        after_context = cls._extract_context(original_text, original_units, i2, "after")

        return f"{before_context or '∅'} 【 {original_fragment or '∅'} / {new_fragment or '∅'} 】{after_context or '∅'}"

    @classmethod
    def _extract_context(
        cls,
        original_text: str,
        original_units: List[str],
        pivot: int,
        direction: str,
        window: int = 5,
    ) -> str:
        if is_mainly_cjk(original_text) and " " not in original_text:
            text = "".join(original_units)
            char_offset = sum(len(token) for token in original_units[:pivot])
            if direction == "before":
                return text[max(0, char_offset - window) : char_offset]
            return text[char_offset : char_offset + window]

        if direction == "before":
            return cls._format_units(original_units[max(0, pivot - window) : pivot], original_text)
        return cls._format_units(original_units[pivot : pivot + window], original_text)

    @classmethod
    def _format_units(cls, units: List[str], text: str) -> str:
        if is_mainly_cjk(text) and " " not in text:
            return "".join(units)
        return " ".join(units)

    @staticmethod
    def _copy_segments(segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        return [
            ASRDataSeg(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            for segment in segments
        ]

    @classmethod
    def _clone_sentence_group(
        cls,
        group: SentenceGroup,
        *,
        text: str,
        optimized_text: str,
        optimize_logs: List[str],
        segments: List[ASRDataSeg],
    ) -> SentenceGroup:
        return SentenceGroup(
            index=group.index,
            segments=segments,
            text=text,
            optimized_text=optimized_text,
            optimize_logs=list(optimize_logs),
        )

    @classmethod
    def _build_insert_segments(
        cls,
        original_segments: List[ASRDataSeg],
        optimized_tokens: List[str],
        insert_at: int,
        token_start: int,
        token_end: int,
    ) -> List[ASRDataSeg]:
        insert_count = token_end - token_start
        if insert_count <= 0:
            return []

        left_anchor = original_segments[insert_at - 1] if insert_at > 0 else None
        right_anchor = (
            original_segments[insert_at] if insert_at < len(original_segments) else None
        )

        if left_anchor and right_anchor and right_anchor.start_time > left_anchor.end_time:
            time_ranges = cls._split_time_range(
                left_anchor.end_time,
                right_anchor.start_time,
                insert_count,
            )
        else:
            anchor = left_anchor or right_anchor
            if anchor is None:
                time_ranges = [(0, 0)] * insert_count
            else:
                time_ranges = [(anchor.start_time, anchor.end_time)] * insert_count

        return [
            ASRDataSeg(
                text=optimized_tokens[token_index],
                start_time=start_time,
                end_time=end_time,
            )
            for token_index, (start_time, end_time) in zip(
                range(token_start, token_end),
                time_ranges,
            )
        ]

    @classmethod
    def _build_replace_segments(
        cls,
        original_segments: List[ASRDataSeg],
        optimized_tokens: List[str],
        original_start: int,
        original_end: int,
        token_start: int,
        token_end: int,
    ) -> List[ASRDataSeg]:
        old_count = original_end - original_start
        new_count = token_end - token_start

        if new_count <= 0:
            return []

        if old_count == new_count and old_count > 0:
            return [
                cls._copy_segment_with_text(
                    original_segments[old_index],
                    optimized_tokens[token_index],
                )
                for old_index, token_index in zip(
                    range(original_start, original_end),
                    range(token_start, token_end),
                )
            ]

        if old_count <= 0:
            return cls._build_insert_segments(
                original_segments=original_segments,
                optimized_tokens=optimized_tokens,
                insert_at=original_start,
                token_start=token_start,
                token_end=token_end,
            )

        time_ranges = cls._split_time_range(
            original_segments[original_start].start_time,
            original_segments[original_end - 1].end_time,
            new_count,
        )
        return [
            ASRDataSeg(
                text=optimized_tokens[token_index],
                start_time=start_time,
                end_time=end_time,
            )
            for token_index, (start_time, end_time) in zip(
                range(token_start, token_end),
                time_ranges,
            )
        ]

    @staticmethod
    def _copy_segment_with_text(segment: ASRDataSeg, text: str) -> ASRDataSeg:
        return ASRDataSeg(
            text=text,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )

    @classmethod
    def _count_punctuation_changes(cls, original_text: str, optimized_text: str) -> int:
        original_tokens = split_tokens(original_text)
        optimized_tokens = split_tokens(optimized_text)
        original_parts = [split_token_parts(token) for token in original_tokens]
        optimized_parts = [split_token_parts(token) for token in optimized_tokens]
        opcodes = Levenshtein.opcodes(
            comparison_bases_from_tokens([part.original for part in original_parts]),
            comparison_bases_from_tokens([part.original for part in optimized_parts]),
        )
        punctuation_changes = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                continue
            for old_index, new_index in zip(range(i1, i2), range(j1, j2)):
                if (
                    original_parts[old_index].original.lower()
                    != optimized_parts[new_index].original.lower()
                ):
                    punctuation_changes += 1
        return punctuation_changes

    @staticmethod
    def _merge_edit_opcodes(
        opcodes: List[Tuple[str, int, int, int, int]],
    ) -> List[Tuple[str, int, int, int, int]]:
        merged: List[Tuple[str, int, int, int, int]] = []
        index = 0
        while index < len(opcodes):
            tag, i1, i2, j1, j2 = opcodes[index]
            if tag == "equal":
                merged.append((tag, i1, i2, j1, j2))
                index += 1
                continue

            merged_i1, merged_i2 = i1, i2
            merged_j1, merged_j2 = j1, j2
            index += 1
            while index < len(opcodes) and opcodes[index][0] != "equal":
                _, next_i1, next_i2, next_j1, next_j2 = opcodes[index]
                merged_i2 = next_i2
                merged_j2 = next_j2
                index += 1

            if merged_i1 == merged_i2:
                merged_tag = "insert"
            elif merged_j1 == merged_j2:
                merged_tag = "delete"
            else:
                merged_tag = "replace"
            merged.append((merged_tag, merged_i1, merged_i2, merged_j1, merged_j2))

        return merged

    @staticmethod
    def _split_time_range(start_time: int, end_time: int, count: int) -> List[Tuple[int, int]]:
        if count <= 0:
            return []
        if count == 1:
            return [(start_time, end_time)]

        total = end_time - start_time
        points = [start_time + (total * index) // count for index in range(count + 1)]
        return list(zip(points[:-1], points[1:]))

    def stop(self) -> None:
        """停止优化器并清理资源"""
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
