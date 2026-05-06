from my_video.cli import output
from my_video.cli.config import get_toml_value, load_toml_config
from my_video.core.prompts import (
    generate_shared_prompt,
    get_prompt_expressiveness,
    get_prompt_faithfulness,
)
from my_video.core.utils import ask_gpt


def _load_config() -> dict:
    config, _ = load_toml_config()
    return config or {}


def valid_translate_result(result: dict, required_keys: list[str], required_sub_keys: list[str]):
    if not all(key in result for key in required_keys):
        missing = sorted(set(required_keys) - set(result.keys()))
        return {"status": "error", "message": f"Missing required key(s): {', '.join(missing)}"}

    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            missing = sorted(set(required_sub_keys) - set(result[key].keys()))
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(missing)}"}

    return {"status": "success", "message": "Translation completed"}


def _log_translation_block(block_name: str, translation_result: dict) -> None:
    output.info(f"{block_name}:")
    for key in translation_result:
        item = translation_result[key]
        output.info(f"  Origin: {item['origin']}")
        output.info(f"  Direct: {item['direct']}")
        free = item.get("free")
        if free:
            output.info(f"  Free:   {free}")


def translate_lines(lines, previous_content_prompt, after_content_prompt, things_to_note_prompt, summary_prompt, index=0):
    shared_prompt = generate_shared_prompt(
        previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt
    )

    def retry_translation(prompt, length, step_name):
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length + 1)], ["direct"])

        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length + 1)], ["free"])

        for retry in range(3):
            if step_name == "faithfulness":
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_faith,
                    log_title=f"translate_{step_name}",
                )
            else:
                result = ask_gpt(
                    prompt + retry * " ",
                    resp_type="json",
                    valid_def=valid_express,
                    log_title=f"translate_{step_name}",
                )

            if len(lines.split("\n")) == len(result):
                return result
            if retry != 2:
                output.warn(f"{step_name.capitalize()} translation of block {index} failed, retrying")

        raise ValueError(
            f"{step_name.capitalize()} translation of block {index} failed after 3 retries. "
            "Please check output/gpt_log/error.json for details."
        )

    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split("\n")), "faithfulness")

    for key in faith_result:
        faith_result[key]["direct"] = faith_result[key]["direct"].replace("\n", " ")

    config = _load_config()
    reflect_translate = bool(get_toml_value(config, "subtitle.translate.reflect", False))
    if not reflect_translate:
        _log_translation_block(f"Translation block {index}", faith_result)
        translate_result = "\n".join([faith_result[i]["direct"].strip() for i in faith_result])
        return translate_result, lines

    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split("\n")), "expressiveness")
    _log_translation_block(f"Translation block {index}", express_result)

    translate_result = "\n".join([express_result[i]["free"].replace("\n", " ").strip() for i in express_result])
    if len(lines.split("\n")) != len(translate_result.split("\n")):
        raise ValueError(f"Translation of block {index} failed due to length mismatch")

    return translate_result, lines
