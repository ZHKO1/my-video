"""翻译器基类"""

import atexit
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from my_video.cli import output
from my_video.core.asr.asr_data import SubtitleLine, SubtitleLines
from my_video.core.translate.types import TargetLanguage
from my_video.core.utils.cache import generate_cache_key, get_translate_cache


class BaseTranslator(ABC):
    """翻译器基类"""

    def __init__(
        self,
        thread_num: int,
        target_language: TargetLanguage,
    ):
        self.thread_num = thread_num
        self.target_language = target_language
        self.is_running = True
        self.executor = None
        self._cache = get_translate_cache()

        self._init_thread_pool()

    def _init_thread_pool(self):
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def translate_subtitle(
        self, subtitle_data: SubtitleLines
    ) -> SubtitleLines:
        """翻译字幕文件并回写翻译结果。"""
        try:
            chunks = self._batch_subtitle_lines(subtitle_data.lines)

            # 多线程翻译
            self._parallel_translate(chunks)
            return subtitle_data
        except Exception as e:
            output.error(f"Translation failed: {e!s}")
            raise RuntimeError(f"Translation failed: {e!s}") from e

    def _parallel_translate(self, chunks: list[list[SubtitleLine]]) -> None:
        """并行翻译All块。"""
        future_to_chunk = {}
        failed_count = 0
        total_segments = sum(len(c) for c in chunks)

        for chunk in chunks:
            future = self.executor.submit(self._safe_translate_chunk, chunk)
            future_to_chunk[future] = chunk

        for future in as_completed(future_to_chunk):
            if not self.is_running:
                break
            try:
                future.result()
            except Exception as e:
                output.error(f"Translation chunk failed: {e}")
                failed_count += len(future_to_chunk[future])

        # Raise if all or most translations failed
        if failed_count > 0 and total_segments > 0:
            fail_rate = failed_count / total_segments
            if fail_rate >= 0.5:
                raise RuntimeError(
                    f"Translation failed: {failed_count}/{total_segments} segments failed "
                    f"({fail_rate:.0%}). Check your API key and network connection."
                )
            elif failed_count > 0:
                output.warn(
                    f"Translation partially failed: {failed_count}/{total_segments} segments"
                )

    def _get_cache_key(self, chunk: list[SubtitleLine]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(
            [{"key": line.line_id, "text": line.text} for line in chunk]
        )
        lang = self.target_language.value
        return f"{class_name}:{chunk_key}:{lang}"

    def _safe_translate_chunk(self, chunk: list[SubtitleLine]) -> list[SubtitleLine]:
        """安全的翻译块"""
        try:
            cache_key = self._get_cache_key(chunk)
            try:
                cached_result = self._cache.get(cache_key, default=None)
            except Exception:
                # Graceful degradation: corrupted cache (e.g. old pickle from app→my_video rename)
                cached_result = None
                self._cache.delete(cache_key)
            if cached_result is not None:
                self._apply_translated_chunk(chunk, cached_result)
                return chunk

            result = self._translate_chunk(chunk)

            self._cache.set(cache_key, result, expire=86400 * 7)
            return result

        except Exception as e:
            output.error(f"Translation failed: {e!s}")
            raise

    @abstractmethod
    def _translate_chunk(
        self, subtitle_chunk: list[SubtitleLine]
    ) -> list[SubtitleLine]:
        """翻译字幕块"""
        pass

    @staticmethod
    def _apply_translated_chunk(
        target_chunk: list[SubtitleLine],
        translated_chunk: list[SubtitleLine],
    ) -> None:
        translated_map = {
            line.line_id: line.translate_text for line in translated_chunk
        }
        for line in target_chunk:
            line.translate_text = translated_map.get(line.line_id, line.translate_text)

    @staticmethod
    def _batch_subtitle_lines(lines: list[SubtitleLine]) -> list[list[SubtitleLine]]:
        grouped: list[list[SubtitleLine]] = []
        current_group_index: int | None = None
        current_group: list[SubtitleLine] = []

        for line in lines:
            if current_group_index is None:
                current_group_index = line.group_index
            if line.group_index != current_group_index:
                grouped.append(current_group)
                current_group = []
                current_group_index = line.group_index
            current_group.append(line)

        if current_group:
            grouped.append(current_group)

        batches: list[list[SubtitleLine]] = []
        current_batch: list[SubtitleLine] = []
        current_group_count = 0
        seen_groups: set[int] = set()

        for group_lines in grouped:
            group_index = group_lines[0].group_index
            if (
                current_batch
                and current_group_count >= 20
                and group_index not in seen_groups
            ):
                batches.append(current_batch)
                current_batch = []
                current_group_count = 0
                seen_groups = set()

            if group_index not in seen_groups:
                current_group_count += 1
                seen_groups.add(group_index)
            current_batch.extend(group_lines)

        if current_batch:
            batches.append(current_batch)

        return batches

    def stop(self):
        """停止翻译器"""
        if not self.is_running:
            return

        self.is_running = False
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                output.error(f"Error closing thread pool: {e!s}")
            finally:
                self.executor = None
