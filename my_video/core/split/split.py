

# ==================== 配置常量 ====================

# 字数限制
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import re
import traceback
from typing import List, Union

from my_video.cli import output
from my_video.core.asr.asr_data import ASRData, ASRDataSeg
from my_video.core.split.split_by_llm import split_by_llm
from my_video.core.utils.text_utils import count_words, is_mainly_cjk, is_pure_punctuation, is_space_separated_language


MAX_WORD_COUNT_CJK = 25  # CJK文本单行最大字数
MAX_WORD_COUNT_ENGLISH = 18  # 英文文本单行最大单词数

# Segments阈值
SEGMENT_WORD_THRESHOLD = 500  # 长文本Segments阈值(字数)

# 时间间隔
MAX_GAP = 1500  # 允许的最大时间间隔(毫秒)
MERGE_SHORT_GAP = 200  # 短Segments合并时间阈值(毫秒)
MERGE_VERY_SHORT_GAP = 500  # 极短Segments合并时间阈值(毫秒)

# 短Segments合并阈值
MERGE_MIN_WORDS = 5  # 短Segments最小字数阈值
MERGE_VERY_SHORT_WORDS = 3  # 极短Segments字数阈值

# 分割相关
SPLIT_SEARCH_RANGE = 30  # 分割点前后搜索范围
TIME_GAP_WINDOW_SIZE = 5  # 时间间隔窗口大小
TIME_GAP_MULTIPLIER = 3  # 大间隔判断倍数
MIN_GROUP_SIZE = 5  # 最小分组大小

# 规则分割
RULE_SPLIT_GAP = 500  # 规则分割时间间隔阈值(毫秒)
RULE_MIN_SEGMENT_SIZE = 4  # 规则分割最小Segments大小

# 常见词分割
PREFIX_WORD_RATIO = 0.6  # 前缀词分割比例
SUFFIX_WORD_RATIO = 0.4  # 后缀词分割比例

# 匹配相关
MATCH_SIMILARITY_THRESHOLD = 0.5  # 文本匹配相似度阈值
MATCH_MAX_SHIFT = 30  # 匹配滑动窗口最大偏移
MATCH_MAX_UNMATCHED = 5  # 允许的最大未匹配句子数
MATCH_LARGE_SHIFT = 100  # 未匹配时的大偏移量

SENTENCE_END_PATTERN = re.compile(r"[?!.？！。]+$")

def preprocess_segments(
    segments: List[ASRDataSeg], need_lower: bool = True
) -> List[ASRDataSeg]:
    """预处理ASRSegments

    1. 移除纯标点符号的Segments
    2. 为需要空格分隔的语言添加空格（英语、俄语、阿拉伯语等，不包括CJK）

    Args:
        segments: ASR数据Segments列表
        need_lower: 是否转小写（仅对拉丁和西里尔字母有效）

    Returns:
        处理后的Segments列表
    """
    new_segments = []
    for seg in segments:
        if not is_pure_punctuation(seg.text):
            text = seg.text.strip()
            # 检查是否为需要空格分隔的语言（不包括CJK）
            if is_space_separated_language(text):
                if need_lower:
                    text = text.lower()
                seg.text = text + " "
            new_segments.append(seg)
    return new_segments

class SubtitleSplitter:
    """字幕智能分割器

    使用LLM进行语义Segments,支持缓存、并发处理和规则降级。
    """

    def __init__(
        self,
        thread_num,
        model,
        max_word_count_cjk: int = MAX_WORD_COUNT_CJK,
        max_word_count_english: int = MAX_WORD_COUNT_ENGLISH,
    ):
        """初始化分割器

        Args:
            thread_num: 并发线程数
            model: LLM模型名称
            max_word_count_cjk: CJK最大字数
            max_word_count_english: 英文最大单词数
        """
        self.thread_num = thread_num
        self.model = model
        self.max_word_count_cjk = max_word_count_cjk
        self.max_word_count_english = max_word_count_english
        self.is_running = True
        self._init_thread_pool()

    def _init_thread_pool(self):
        """初始化线程池并注册清理"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)
    
    def split_subtitle(self, subtitle_data: Union[str, ASRData]):
        # 1. Reading字幕
        if isinstance(subtitle_data, str):
            asr_data = ASRData.from_subtitle_file(subtitle_data)
        else:
            asr_data = subtitle_data

        asr_data.segments = preprocess_segments(asr_data.segments, need_lower=False)
        # txt = asr_data.to_txt().replace("\n", "")
        # output.info(f"total text length: {len(txt)}")

        # 2. 按照批次划分
        sentence_groups = self._split_segments_by_sentence(asr_data.segments)
        batched_segments = self._batch_sentence_groups(sentence_groups)
        asr_data_list = [ASRData(batch) for batch in batched_segments if batch]

        # 4. 并发处理
        processed_segments = self._process_segments(asr_data_list)
        
        # 5. 合并
        final_segments = self._merge_processed_segments(processed_segments)

        return ASRData(final_segments)


    def _split_segments_by_sentence(
        self, segments: List[ASRDataSeg]
    ) -> List[List[ASRDataSeg]]:
        """按句末标点切分，但保留原始词级Segments。"""
        sentence_groups: List[List[ASRDataSeg]] = []
        current_group: List[ASRDataSeg] = []

        for seg in segments:
            current_group.append(seg)
            if SENTENCE_END_PATTERN.search(seg.text.strip()):
                sentence_groups.append(current_group)
                current_group = []

        if current_group:
            sentence_groups.append(current_group)

        return sentence_groups

    def _batch_sentence_groups(
        self,
        sentence_groups: List[List[ASRDataSeg]],
        threshold: int = SEGMENT_WORD_THRESHOLD,
    ) -> List[List[ASRDataSeg]]:
        """将完整句按批次分组，每批总字数不超过阈值。"""
        batches: List[List[ASRDataSeg]] = []
        current_batch: List[ASRDataSeg] = []
        current_count = 0

        for sentence_group in sentence_groups:
            sentence_text = "".join(seg.text for seg in sentence_group)
            sentence_count = count_words(sentence_text)

            if current_batch and current_count + sentence_count > threshold:
                batches.append(current_batch)
                current_batch = []
                current_count = 0

            current_batch.extend(sentence_group)
            current_count += sentence_count

        if current_batch:
            batches.append(current_batch)

        return batches

    def _group_by_time_gaps(
        self,
        segments: List[ASRDataSeg],
        max_gap: int = MAX_GAP,
        check_large_gaps: bool = False,
    ) -> List[List[ASRDataSeg]]:
        """Grouped by time gaps

        Args:
            segments: Segments列表
            max_gap: 最大允许间隔(ms)
            check_large_gaps: 是否检查异常大间隔

        Returns:
            分组后的列表
        """
        if not segments:
            return []

        result = []
        current_group = [segments[0]]
        recent_gaps = []

        for i in range(1, len(segments)):
            time_gap = segments[i].start_time - segments[i - 1].end_time

            # 检查异常大间隔
            if check_large_gaps:
                recent_gaps.append(time_gap)
                if len(recent_gaps) > TIME_GAP_WINDOW_SIZE:
                    recent_gaps.pop(0)
                if len(recent_gaps) == TIME_GAP_WINDOW_SIZE:
                    avg_gap = sum(recent_gaps) / len(recent_gaps)
                    if (
                        time_gap > avg_gap * TIME_GAP_MULTIPLIER
                        and len(current_group) > MIN_GROUP_SIZE
                    ):
                        result.append(current_group)
                        current_group = []
                        recent_gaps = []

            # 超过最大间隔则分组
            if time_gap > max_gap:
                result.append(current_group)
                current_group = []
                recent_gaps = []

            current_group.append(segments[i])

        if current_group:
            result.append(current_group)

        return result
    
    def _split_long_segment(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """拆分超长Segments

        策略:寻找最大时间间隔点进行拆分

        Args:
            segments: Segments列表

        Returns:
            拆分后的Segments列表
        """
        result_segs = []
        segments_to_process = [segments]

        while segments_to_process:
            current_segments = segments_to_process.pop(0)

            if not current_segments:
                continue

            merged_text = "".join(seg.text for seg in current_segments)
            max_word_count = (
                self.max_word_count_cjk
                if is_mainly_cjk(merged_text)
                else self.max_word_count_english
            )
            n = len(current_segments)

            # Segments足够短或无法继续拆分
            if count_words(merged_text) <= max_word_count or n < RULE_MIN_SEGMENT_SIZE:
                merged_seg = ASRDataSeg(
                    merged_text.strip(),
                    current_segments[0].start_time,
                    current_segments[-1].end_time,
                )
                result_segs.append(merged_seg)
                continue

            # 检查时间间隔
            gaps = [
                current_segments[i + 1].start_time - current_segments[i].end_time
                for i in range(n - 1)
            ]
            all_equal = all(abs(gap - gaps[0]) < 1e-6 for gap in gaps)

            if all_equal:
                # 间隔相等:中间分割
                split_index = n // 2
            else:
                # 间隔不等:寻找最大间隔点
                start_idx = max(n // 6, 1)
                end_idx = min((5 * n) // 6, n - 2)
                split_index = max(
                    range(start_idx, end_idx),
                    key=lambda i: current_segments[i + 1].start_time
                    - current_segments[i].end_time,
                    default=n // 2,
                )
                if split_index == 0 or split_index == n - 1:
                    split_index = n // 2

            # 分割并加入处理队列
            first_segs = current_segments[: split_index + 1]
            second_segs = current_segments[split_index + 1 :]
            segments_to_process.extend([first_segs, second_segs])

        # 按时间排序
        result_segs.sort(key=lambda seg: seg.start_time)
        return result_segs

    def _process_segments(self, asr_data_list: List[ASRData]) -> List[List[ASRDataSeg]]:
        """并发处理AllSegments"""
        futures = []
        for asr_data in asr_data_list:
            if not self.executor:
                raise ValueError("Thread pool not initialized")
            future = self.executor.submit(self._process_single_segment, asr_data)
            futures.append(future)

        processed_segments = []
        for future in as_completed(futures):
            if not self.is_running:
                break
            try:
                result = future.result()
                processed_segments.append(result)
            except Exception as e:
                output.error(traceback.format_exc())
                output.error(f"Segment processing failed:{str(e)}")

        return processed_segments

    def _process_single_segment(self, asr_data_part: ASRData) -> List[ASRDataSeg]:
        """处理单个Segments(带重试和降级)"""
        if not asr_data_part.segments:
            return []
        try:
            return self._process_by_llm(asr_data_part.segments)
        except Exception as e:
            output.warn(f"LLM processing failed, falling back to rules: {str(e)}")
            raise

    def _process_by_llm(self, segments: List[ASRDataSeg]) -> List[ASRDataSeg]:
        """使用LLM进行智能Segments

        Args:
            segments: ASRSegments列表

        Returns:
            处理后的Segments列表
        """
        txt = "".join([seg.text for seg in segments])
        output.info(f"Calling API for segmentation,text length: {count_words(txt)}")

        sentences = split_by_llm(
            text=txt,
            model=self.model,
            max_word_count_cjk=self.max_word_count_cjk,
            max_word_count_english=self.max_word_count_english,
        )

        return self._merge_segments_based_on_sentences(segments, sentences)

    def _merge_segments_based_on_sentences(
        self,
        segments: List[ASRDataSeg],
        sentences: List[str],
        max_unmatched: int = MATCH_MAX_UNMATCHED,
    ) -> List[ASRDataSeg]:
        """基于LLM返回的句子列表合并ASRSegments

        使用滑动窗口匹配算法:
        1. 对每个LLM句子,寻找最佳匹配的ASRSegments序列
        2. 使用相似度算法进行匹配
        3. 合并匹配的Segments

        Args:
            segments: ASRSegments列表
            sentences: LLM返回的句子列表
            max_unmatched: 允许的最大未匹配句子数

        Returns:
            合并后的Segments列表

        Raises:
            ValueError: Unmatched sentences exceeded threshold
        """

        def preprocess_text(s: str) -> str:
            """文本标准化:小写+空格规范化"""
            return " ".join(s.lower().split())

        asr_texts = [seg.text for seg in segments]
        asr_len = len(asr_texts)
        asr_index = 0
        threshold = MATCH_SIMILARITY_THRESHOLD
        max_shift = MATCH_MAX_SHIFT
        unmatched_count = 0

        new_segments = []

        for sentence in sentences:
            output.info("==========")
            output.info(f"Processing sentence: {sentence}")
            output.info("Next sentences: :" + "".join(asr_texts[asr_index : asr_index + 10]))

            sentence_proc = preprocess_text(sentence)
            word_count = count_words(sentence_proc)
            best_ratio = 0.0
            best_pos = None
            best_window_size = 0

            # 滑动窗口大小
            max_window_size = min(word_count * 2, asr_len - asr_index)
            min_window_size = max(1, word_count // 2)
            window_sizes = sorted(
                range(min_window_size, max_window_size + 1),
                key=lambda x: abs(x - word_count),
            )

            # 滑动窗口匹配
            for window_size in window_sizes:
                max_start = min(asr_index + max_shift + 1, asr_len - window_size + 1)
                for start in range(asr_index, max_start):
                    substr = "".join(asr_texts[start : start + window_size])
                    substr_proc = preprocess_text(substr)
                    ratio = difflib.SequenceMatcher(
                        None, sentence_proc, substr_proc
                    ).ratio()

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = start
                        best_window_size = window_size
                    if ratio == 1.0:
                        break
                if best_ratio == 1.0:
                    break

            # 处理匹配结果
            if best_ratio >= threshold and best_pos is not None:
                start_seg_index = best_pos
                end_seg_index = best_pos + best_window_size - 1

                segs_to_merge = segments[start_seg_index : end_seg_index + 1]

                # 按时间切分避免跨度过大
                seg_groups = self._group_by_time_gaps(segs_to_merge, max_gap=MAX_GAP)

                for group in seg_groups:
                    merged_text = "".join(seg.text for seg in group)
                    merged_start_time = group[0].start_time
                    merged_end_time = group[-1].end_time
                    merged_seg = ASRDataSeg(
                        merged_text, merged_start_time, merged_end_time
                    )

                    output.info(f"Merged segments: {merged_seg.text}")

                    # 拆分超长Segments
                    split_segs = self._split_long_segment(group)
                    new_segments.extend(split_segs)

                max_shift = MATCH_MAX_SHIFT
                asr_index = end_seg_index + 1
            else:
                output.warn(f"Cannot match sentence: {sentence}")
                unmatched_count += 1
                if unmatched_count > max_unmatched:
                    raise ValueError(f"Unmatched sentences exceeded threshold {max_unmatched},processing aborted")
                max_shift = MATCH_LARGE_SHIFT
                asr_index = min(asr_index + 1, asr_len - 1)

        return new_segments

    def _merge_processed_segments(
        self, processed_segments: List[List[ASRDataSeg]]
    ) -> List[ASRDataSeg]:
        """合并All处理后的Segments并排序"""
        final_segments = []
        for segments in processed_segments:
            final_segments.extend(segments)

        final_segments.sort(key=lambda seg: seg.start_time)
        return final_segments

    def stop(self):
        """停止分割器并清理资源"""
        if not self.is_running:
            return
        self.is_running = False
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                output.error(e)
                output.error(traceback.format_exc())
            finally:
                self.executor = None
