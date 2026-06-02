"""LLM 翻译器（使用 OpenAI）"""

import json
from typing import Any, Dict, Tuple

import json_repair
import openai

from my_video.cli import output
from my_video.core.asr.asr_data import SubtitleLine
from my_video.core.llm import call_llm
from my_video.core.prompts import get_prompt
from my_video.core.translate.base import BaseTranslator
from my_video.core.translate.types import TargetLanguage
from my_video.core.utils.cache import generate_cache_key


class LLMTranslator(BaseTranslator):
    """LLM 翻译器（OpenAI兼容API）"""

    MAX_STEPS = 3

    def __init__(
        self,
        thread_num: int,
        target_language: TargetLanguage,
        model: str,
        custom_prompt: str,
        is_reflect: bool,
    ):
        super().__init__(thread_num=thread_num, target_language=target_language)
        self.model = model
        self.custom_prompt = custom_prompt
        self.is_reflect = is_reflect

    def _get_translate_prompt_info(self) -> Tuple[str, str]:
        prompt_path = "translate/reflect" if self.is_reflect else "translate/standard"
        prompt = get_prompt(
            prompt_path,
            target_language=self.target_language,
            custom_prompt=self.custom_prompt,
        )
        return prompt_path, prompt

    def _translate_chunk(self, subtitle_chunk: list[SubtitleLine]) -> list[SubtitleLine]:
        output.info(
            f"[+]正在翻译字幕: {subtitle_chunk[0].group_index} - {subtitle_chunk[-1].group_index}"
        )

        subtitle_dict = {line.line_id: line.text for line in subtitle_chunk}
        _, prompt = self._get_translate_prompt_info()

        try:
            result_dict = self._agent_loop(prompt, subtitle_dict)
            processed_result = self._normalize_result(result_dict)
            for line in subtitle_chunk:
                line.translate_text = processed_result.get(line.line_id, line.text)
            return subtitle_chunk
        except openai.RateLimitError as e:
            output.error(f"OpenAI Rate Limit Error: {str(e)}")
            raise
        except openai.AuthenticationError as e:
            output.error(f"OpenAI Authentication Error: {str(e)}")
            raise
        except openai.NotFoundError as e:
            output.error(f"OpenAI NotFound Error: {str(e)}")
            raise
        except Exception as e:
            output.error(f"LLM translation error: {e}")
            raise

    def _normalize_result(self, result_dict: Dict[str, Any]) -> Dict[str, str]:
        if self.is_reflect:
            return {
                key: str(value.get("native_translation", value))
                if isinstance(value, dict)
                else str(value)
                for key, value in result_dict.items()
            }
        return {key: str(value) for key, value in result_dict.items()}

    def _agent_loop(
        self, system_prompt: str, subtitle_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_payload(subtitle_dict)},
        ]
        last_response_dict: Dict[str, Any] | None = None

        for _ in range(self.MAX_STEPS):
            response = call_llm(messages=messages, model=self.model)
            response_dict = json_repair.loads(response.choices[0].message.content.strip())
            last_response_dict = response_dict
            is_valid, error_message = self._validate_llm_response(
                response_dict, subtitle_dict
            )
            if is_valid:
                return response_dict

            messages.append(
                {
                    "role": "assistant",
                    "content": self._format_payload(response_dict),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {error_message}\n\n"
                        f"Fix the errors above and output ONLY a valid JSON dictionary with ALL {len(subtitle_dict)} keys"
                    ),
                }
            )

        return last_response_dict or {}

    def _validate_llm_response(
        self, response_dict: Any, subtitle_dict: Dict[str, str]
    ) -> Tuple[bool, str]:
        if not isinstance(response_dict, dict):
            return (
                False,
                f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0:0': 'text', '0:1': 'text'}}",
            )

        expected_keys = set(subtitle_dict.keys())
        actual_keys = set(response_dict.keys())

        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            error_parts = []
            if missing:
                error_parts.append(f"Missing keys {missing}")
            if extra:
                error_parts.append(f"Extra keys {extra}")
            return False, "; ".join(error_parts)

        if self.is_reflect:
            for key, value in response_dict.items():
                if not isinstance(value, dict):
                    return False, f"Key '{key}': value must be a dict with 'native_translation' field."
                if "native_translation" not in value:
                    return False, f"Key '{key}': missing 'native_translation' field."

        return True, ""

    def _format_payload(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _get_cache_key(self, chunk: list[SubtitleLine]) -> str:
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(
            [{"key": line.line_id, "text": line.text} for line in chunk]
        )
        lang = self.target_language.value
        model = self.model
        prompt_path, prompt = self._get_translate_prompt_info()
        prompt_key = generate_cache_key(
            {
                "prompt_path": prompt_path,
                "prompt": prompt,
            }
        )
        return f"{class_name}:{chunk_key}:{lang}:{model}:{prompt_key}"
