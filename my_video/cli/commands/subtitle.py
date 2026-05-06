"""subtitle command — lightweight placeholder for subtitle processing."""

import concurrent.futures
import math
from argparse import Namespace
from difflib import SequenceMatcher

from my_video.cli import output
from my_video.cli import exit_codes as EXIT
from my_video.cli.config import get_toml_value, get_work_dir
from my_video.core.spacy_utils import (
    init_nlp,
    split_by_comma_main,
    split_by_mark,
    split_long_by_root_main,
    split_sentences_main,
)
from my_video.core.prompts import get_split_prompt
from my_video.core.spacy_utils.load_nlp_model import resolve_spacy_language
from my_video.core.utils import check_file_exists, get_joiner
from my_video.core.utils.models import OutputPaths, build_output_paths


@check_file_exists(lambda paths, *_args, **_kwargs: paths.split_by_nlp)
def split_by_spacy(paths: OutputPaths):
    nlp = init_nlp()
    split_by_mark(nlp, paths)
    split_by_comma_main(nlp, paths)
    split_sentences_main(nlp, paths)
    split_long_by_root_main(nlp, paths)


def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]


def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split("[br]")
    start = 0
    language = resolve_spacy_language()
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            output.warn(f"Low similarity found at the best split point: {max_similarity:.3f}")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            output.warn(f"Unable to find a suitable split point for part {i + 1}.")

    return split_positions


def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0):
    """Split a long sentence using GPT and return the result as a string."""
    try:
        from my_video.core.utils.ask_gpt import ask_gpt
    except ImportError as e:
        raise RuntimeError(
            "Split-by-meaning requires GPT prompt/config utilities that are not available yet."
        ) from e

    split_prompt = get_split_prompt(sentence, num_parts, word_limit)

    def valid_split(response_data):
        choice = response_data["choice"]
        if f"split{choice}" not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        if "[br]" not in response_data[f"split{choice}"]:
            return {"status": "error", "message": "Split failed, no [br] found"}
        return {"status": "success", "message": "Split completed"}

    response_data = ask_gpt(
        split_prompt + " " * retry_attempt,
        resp_type="json",
        valid_def=valid_split,
        log_title="split_by_meaning",
    )
    choice = response_data["choice"]
    best_split = response_data[f"split{choice}"]
    split_points = find_split_positions(sentence, best_split)
    # Split the sentence based on the inferred split points.
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + "\n" + sentence[split_point:]
        else:
            parts = best_split.split("\n")
            last_part = parts[-1]
            offset = split_point - split_points[i - 1]
            parts[-1] = last_part[:offset] + "\n" + last_part[offset:]
            best_split = "\n".join(parts)

    if index != -1:
        output.success(f"Sentence {index} has been successfully split")
    output.info(f"Original: {sentence}")
    output.info(f"Split: {best_split.replace(chr(10), ' || ')}")

    return best_split


def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            tokens = tokenize_sentence(sentence, nlp)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(
                    split_sentence,
                    sentence,
                    num_parts,
                    max_length,
                    index=index,
                    retry_attempt=retry_attempt,
                )
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split("\n")
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]


@check_file_exists(lambda paths, *_args, **_kwargs: paths.split_by_meaning)
def split_sentences_by_meaning(paths: OutputPaths, config: dict):
    """The main function to split sentences by meaning."""
    with paths.split_by_nlp.open("r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    max_length = get_toml_value(config, "subtitle.max_split_length", 20)
    max_workers = get_toml_value(config, "subtitle.max_workers", 4)
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(
            sentences,
            max_length=max_length,
            max_workers=max_workers,
            nlp=nlp,
            retry_attempt=retry_attempt,
        )

    with paths.split_by_meaning.open("w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    output.success("All sentences have been successfully split")

def run(args: Namespace, config: dict) -> int:
    work_dir = get_work_dir(config) or "."
    paths = build_output_paths(work_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    quiet = getattr(args, "quiet", False)

    try:
        split_by_spacy(paths)
        if get_toml_value(config, "subtitle.split_by_meaning", False):
            split_sentences_by_meaning(paths, config)
    except Exception as e:
        output.error(str(e))
        return EXIT.RUNTIME_ERROR

    if quiet:
        print(paths.split_by_nlp)
    return EXIT.SUCCESS
