from my_video.core.split.split import SubtitleSplitter


class TestSubtitleSplitter:
    def test_agent_loop_warns_with_response_id_on_validation_failure(
        self, monkeypatch
    ) -> None:
        splitter = SubtitleSplitter(
            thread_num=1,
            batch_num=20,
            model="test-model",
            custom_prompt="",
            max_word_count=10,
        )
        warnings: list[str] = []

        class DummyMessage:
            content = '{"0":"hello<br>world"}'

        class DummyChoice:
            message = DummyMessage()

        class DummyResponse:
            id = "split-resp-1"
            choices = [DummyChoice()]

        monkeypatch.setattr(
            "my_video.core.split.split.call_llm",
            lambda **kwargs: DummyResponse(),
        )
        monkeypatch.setattr(
            "my_video.core.split.split.output.warn",
            lambda message: warnings.append(message),
        )
        monkeypatch.setattr(
            splitter,
            "_validate_split_result",
            lambda *args, **kwargs: (
                len(warnings) > 0,
                "too many words" if not warnings else "",
            ),
        )

        result = splitter._agent_loop({"0": "hello world"})

        assert result == {0: ["hello", "world"]}
        assert (
            "分割验证失败[split-resp-1]，开始反馈循环 (第1次尝试): too many words"
            in warnings
        )
        splitter.stop()
