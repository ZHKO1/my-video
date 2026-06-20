"""字幕优化模块

使用LLM优化字幕内容，支持agent loop自动验证和修正。
"""

import atexit
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import json_repair
from rapidfuzz.distance import Levenshtein

from my_video.cli import output
from my_video.core.utils.text_diff import (
    render_inline_diff,
    rewrite_segments_with_timestamps,
)
from my_video.core.utils.text_utils import count_words

from ..asr.asr_data import (
    SubtitleSegment,
    SubtitleSegments,
    SubtitleSentence,
    SubtitleSentences,
)
from ..llm import call_llm, get_response_id
from ..prompts import get_prompt
from ..utils.helper import (
    split_tokens,
    text_bases,
)

MAX_STEPS = 3
REFERENCE_TIME_PADDING_MS = 5000


@dataclass
class OptimizationBatch:
    groups: list[SubtitleSentence]
    subtitle_chunk: dict[str, str]
    reference_text: str
    start_time_ms: int
    end_time_ms: int


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
        self.executor: ThreadPoolExecutor | None = None
        self._init_thread_pool()

    def _init_thread_pool(self) -> None:
        """初始化线程池并注册清理函数"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def optimize_subtitle(
        self,
        subtitle_data: SubtitleSentences,
        reference_data: SubtitleSegments | None = None,
    ) -> SubtitleSentences:
        """
        优化字幕句组。
        将优化结果写回输入对象，同时返回一个干净的副本（optimized_text 和 optimize_log 为空）
        """
        try:
            sentence_groups = subtitle_data.sentences
            batches = self._batch_sentence_groups(sentence_groups, reference_data)

            # 并行优化
            optimized_dict = self._parallel_optimize(batches)

            return self._write_back_groups(sentence_groups, optimized_dict)

        except Exception as e:
            output.error(f"Optimization failed: {e!s}")
            raise RuntimeError(f"Optimization failed: {e!s}") from e

    def _batch_sentence_groups(
        self,
        groups: list[SubtitleSentence],
        reference_data: SubtitleSegments | None = None,
    ) -> list[OptimizationBatch]:
        batches: list[OptimizationBatch] = []
        for index in range(0, len(groups), self.batch_num):
            batch_groups = groups[index : index + self.batch_num]
            if not batch_groups:
                continue
            start_time_ms = batch_groups[0].segments[0].start_time
            end_time_ms = batch_groups[-1].segments[-1].end_time
            subtitle_chunk = {str(group.index): group.text for group in batch_groups}
            batches.append(
                OptimizationBatch(
                    groups=batch_groups,
                    subtitle_chunk=subtitle_chunk,
                    reference_text=self._build_reference_text(
                        reference_data,
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                    ),
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )
            )
        return batches

    @classmethod
    def _build_reference_text(
        cls,
        reference_data: SubtitleSegments | None,
        *,
        start_time_ms: int,
        end_time_ms: int,
    ) -> str:
        if reference_data is None:
            return ""

        window_start = start_time_ms - REFERENCE_TIME_PADDING_MS
        window_end = end_time_ms + REFERENCE_TIME_PADDING_MS
        matched_segments = [
            segment.text.strip()
            for segment in reference_data.segments
            if segment.text.strip()
            and segment.start_time <= window_end
            and segment.end_time >= window_start
        ]
        return " ".join(matched_segments)

    def _parallel_optimize(self, batches: list[OptimizationBatch]) -> dict[str, str]:
        """并行优化All批次

        Args:
            batches: 字幕批次列表

        Returns:
            优化后的字幕字典
        """
        if not self.executor:
            raise ValueError("Thread pool not initialized")

        futures = []
        optimized_dict: dict[str, str] = {}

        # 提交All任务
        for batch in batches:
            future = self.executor.submit(self._optimize_chunk, batch)
            futures.append((future, batch))

        # 收集结果
        for future, batch in futures:
            if not self.is_running:
                break

            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                output.error(f"Optimization batch failed: {e!s}")
                optimized_dict.update(batch.subtitle_chunk)  # 失败时保留原文

        return optimized_dict

    def _optimize_chunk(self, batch: OptimizationBatch) -> dict[str, str]:
        """优化单个字幕批次

        Args:
            batch: 字幕批次数据

        Returns:
            优化后的字幕批次
        """
        output.info(
            f"[+]Optimizing subtitles: {batch.groups[0].index} - "
            f"{batch.groups[-1].index}"
        )
        try:
            result = self.agent_loop(batch.subtitle_chunk, batch.reference_text)
            return result
        except Exception as e:
            output.error(f"Optimization failed: {e!s}")
            return batch.subtitle_chunk

    def agent_loop(
        self,
        subtitle_chunk: dict[str, str],
        reference_text: str,
    ) -> dict[str, str]:
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
            f"<input_subtitle>{json.dumps(subtitle_chunk, ensure_ascii=False)}</input_subtitle>"
        )
        reference_parts = [
            part.strip()
            for part in [reference_text, self.custom_prompt]
            if part.strip()
        ]
        if reference_parts:
            user_prompt += (
                "\n<reference>\n" + "\n".join(reference_parts) + "\n</reference>"
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
            response_id = get_response_id(response)

            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("LLM returned empty result")

            # 解析结果
            parsed_result = json_repair.loads(result_text)
            if not isinstance(parsed_result, dict):
                raise ValueError(
                    f"LLM返回结果类型Error，期望dict，实际{type(parsed_result)}"
                )

            result_dict: dict[str, str] = parsed_result
            last_result = result_dict

            # 验证结果
            is_valid, error_message = self._validate_optimization_result(
                original_chunk=subtitle_chunk, optimized_chunk=result_dict
            )

            if is_valid:
                return result_dict

            # 验证失败，添加反馈
            output.warn(
                f"优化验证失败[{response_id}]，开始反馈循环 (第{step + 1}次尝试): {error_message}"
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
        self, original_chunk: dict[str, str], optimized_chunk: dict[str, str]
    ) -> tuple[bool, str]:
        """验证优化结果

        检查:
        1. 键是否完全匹配
        2. 改动（去掉标点符号）是否过大（相似度 < 0.7）

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

            original_bases = text_bases(original_text)
            optimized_bases = text_bases(optimized_text)
            similarity = Levenshtein.normalized_similarity(
                original_bases,
                optimized_bases,
            )
            similarity_threshold = 0.3 if count_words(original_text) <= 10 else 0.6

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
        sentence_groups: list[SubtitleSentence],
        optimized_dict: dict[str, str],
    ) -> SubtitleSentences:
        new_groups: list[SubtitleSentence] = []
        for group in sentence_groups:
            optimized_text = optimized_dict.get(str(group.index))
            if optimized_text is None or not optimized_text.strip():
                group.optimized_text = ""
                group.optimize_log = ""
                new_groups.append(
                    cls._clone_sentence_group(
                        group,
                        text=group.text,
                        optimized_text="",
                        optimize_log="",
                        segments=cls._copy_segments(group.segments),
                    )
                )
                continue

            if optimized_text.strip() == group.text.strip():
                group.optimized_text = ""
                group.optimize_log = ""
                new_groups.append(
                    cls._clone_sentence_group(
                        group,
                        text=group.text,
                        optimized_text="",
                        optimize_log="",
                        segments=cls._copy_segments(group.segments),
                    )
                )
                continue

            optimize_log = cls._build_group_change_log(group.text, optimized_text)
            rewritten_segments = cls._rewrite_group_segments(
                group.segments, optimized_text
            )
            group.optimized_text = optimized_text
            group.optimize_log = optimize_log
            new_groups.append(
                cls._clone_sentence_group(
                    group,
                    text=optimized_text,
                    optimized_text="",
                    optimize_log="",
                    segments=rewritten_segments,
                )
            )
        return SubtitleSentences(new_groups)

    @classmethod
    def _rewrite_group_segments(
        cls,
        original_segments: list[SubtitleSegment],
        optimized_text: str,
    ) -> list[SubtitleSegment]:
        return rewrite_segments_with_timestamps(
            original_segments,
            optimized_text,
            mode="strict",
        )

    @classmethod
    def _build_group_change_log(cls, original_text: str, optimized_text: str) -> str:
        log = render_inline_diff(original_text, optimized_text)
        return "" if log == " ".join(split_tokens(optimized_text.strip())) else log

    @staticmethod
    def _copy_segments(segments: list[SubtitleSegment]) -> list[SubtitleSegment]:
        return [
            SubtitleSegment(
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
            for segment in segments
        ]

    @classmethod
    def _clone_sentence_group(
        cls,
        group: SubtitleSentence,
        *,
        text: str,
        optimized_text: str,
        optimize_log: str,
        segments: list[SubtitleSegment],
    ) -> SubtitleSentence:
        return SubtitleSentence(
            index=group.index,
            segments=segments,
            text=text,
            optimized_text=optimized_text,
            optimize_log=optimize_log,
        )

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
