from my_video.core.asr.asr_data import ASRDataSeg, ASRSentenceData, SentenceGroup
from my_video.core.translate.base import BaseTranslator
from my_video.core.translate.types import TargetLanguage


class StubTranslator(BaseTranslator):
    def _translate_chunk(self, subtitle_chunk: list[SentenceGroup]) -> list[SentenceGroup]:
        for group in subtitle_chunk:
            group.translate_text = f"ZH:{group.text}"
        return subtitle_chunk


def test_translate_subtitle_writes_to_sentence_group_translate_text() -> None:
    translator = StubTranslator(
        thread_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    )
    sentence_data = ASRSentenceData(
        [
            SentenceGroup(index=0, segments=[ASRDataSeg("hello", 0, 100)], text="hello"),
            SentenceGroup(index=1, segments=[ASRDataSeg("world", 100, 200)], text="world"),
        ]
    )

    translator.translate_subtitle(sentence_data)

    assert [group.translate_text for group in sentence_data.sentences] == ["ZH:hello", "ZH:world"]
    translator.stop()


def test_translate_subtitle_batches_sentence_groups_by_threshold() -> None:
    translator = StubTranslator(
        thread_num=1,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
    )
    large_a = " ".join(["one"] * 300)
    large_b = " ".join(["two"] * 300)
    small_c = "three four"
    groups = [
        SentenceGroup(index=0, segments=[ASRDataSeg("one", 0, 100)], text=large_a),
        SentenceGroup(index=1, segments=[ASRDataSeg("two", 100, 200)], text=large_b),
        SentenceGroup(index=2, segments=[ASRDataSeg("three", 200, 300)], text=small_c),
    ]

    chunks = translator._batch_sentence_groups(groups)

    assert [[group.index for group in chunk] for chunk in chunks] == [[0], [1, 2]]
    translator.stop()
