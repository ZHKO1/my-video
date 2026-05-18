"""subtitle command — lightweight placeholder for subtitle processing."""

import concurrent.futures
import json
import math
from argparse import Namespace
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

import autocorrect_py as autocorrect
import pandas as pd

from my_video.core.utils import except_handler
from my_video.cli import output
from my_video.cli import exit_codes as EXIT
from my_video.cli.config import get_toml_str, get_toml_value, get_work_dir
from my_video.core.prompts import get_align_prompt, get_split_prompt, get_summary_prompt
from my_video.core.subtitle_alignment import align_timestamp
from my_video.core.subtitle_trim import check_len_then_trim
from my_video.core.spacy_utils.load_nlp_model import resolve_spacy_language
from my_video.core.spacy_utils import (
    init_nlp,
    split_by_comma_main,
    split_by_mark,
    split_long_by_root_main,
    split_sentences_main,
)
from my_video.core.translate_lines import translate_lines
from my_video.core.utils import ask_gpt, check_file_exists, get_joiner
from my_video.core.utils.models import OutputPaths, build_output_paths

SUBTITLE_OUTPUT_CONFIGS = [
    ("src.srt", ["Source"]),
    ("trans.srt", ["Translation"]),
    ("src_trans.srt", ["Source", "Translation"]),
    ("trans_src.srt", ["Translation", "Source"]),
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ("src_subs_for_audio.srt", ["Source"]),
    ("trans_subs_for_audio.srt", ["Translation"]),
]


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

CUSTOM_TERMS_PATH = Path("custom_terms.xlsx")


def _get_summary_length(config: dict) -> int:
    value = get_toml_value(config, "subtitle.summary_length", 8000)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 8000


def combine_chunks(paths: OutputPaths, config: dict) -> str:
    """Combine the text chunks identified by whisper into a single long text"""
    source_path = paths.split_by_meaning if paths.split_by_meaning.exists() else paths.split_by_nlp
    with source_path.open("r", encoding="utf-8") as file:
        sentences = file.readlines()
    cleaned_sentences = [line.strip() for line in sentences]
    combined_text = " ".join(cleaned_sentences)
    return combined_text[: _get_summary_length(config)]


def search_things_to_note_in_prompt(paths: OutputPaths, sentence: str):
    """Search for terms to note in the given sentence"""
    if not paths.terminology.exists():
        return None

    with paths.terminology.open("r", encoding="utf-8") as file:
        things_to_note = json.load(file)
    things_to_note_list = [term["src"] for term in things_to_note["terms"] if term["src"].lower() in sentence.lower()]
    if things_to_note_list:
        prompt = "\n".join(
            f'{i+1}. "{term["src"]}": "{term["tgt"]}",'
            f' meaning: {term["note"]}'
            for i, term in enumerate(things_to_note["terms"])
            if term["src"] in things_to_note_list
        )
        return prompt
    return None


def _load_custom_terms() -> tuple[pd.DataFrame | None, dict]:
    if not CUSTOM_TERMS_PATH.exists():
        return None, {"terms": []}

    custom_terms = pd.read_excel(CUSTOM_TERMS_PATH)
    custom_terms_json = {
        "terms": [
            {
                "src": str(row.iloc[0]),
                "tgt": str(row.iloc[1]),
                "note": str(row.iloc[2]),
            }
            for _, row in custom_terms.iterrows()
        ]
    }
    return custom_terms, custom_terms_json


@check_file_exists(lambda paths, *_args, **_kwargs: paths.terminology)
def get_summary(paths: OutputPaths, config: dict):
    """Summarize split subtitles and extract terminology."""
    src_content = combine_chunks(paths, config)
    custom_terms, custom_terms_json = _load_custom_terms()

    if custom_terms is not None and len(custom_terms) > 0:
        output.info(f"Custom terms loaded: {len(custom_terms)} terms")

    summary_prompt = get_summary_prompt(src_content, custom_terms_json)
    output.info("Summarizing and extracting terminology")

    def valid_summary(response_data):
        required_keys = {"src", "tgt", "note"}
        if "terms" not in response_data:
            return {"status": "error", "message": "Invalid response format"}
        for term in response_data["terms"]:
            if not all(key in term for key in required_keys):
                return {"status": "error", "message": "Invalid response format"}
        return {"status": "success", "message": "Summary completed"}

    summary = ask_gpt(summary_prompt, resp_type="json", valid_def=valid_summary, log_title="summary")
    summary["terms"].extend(custom_terms_json["terms"])

    with paths.terminology.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    output.success(f"Summary log saved to {paths.terminology}")


# Function to split text into chunks
def split_chunks_by_chars(paths: OutputPaths, chunk_size, max_i):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    source_path = paths.split_by_meaning if paths.split_by_meaning.exists() else paths.split_by_nlp
    with source_path.open("r", encoding="utf-8") as file:
        sentences = file.read().strip().split("\n")

    chunks = []
    chunk = ""
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + "\n") > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + "\n"
            sentence_count = 1
        else:
            chunk += sentence + "\n"
            sentence_count += 1
    if chunk.strip():
        chunks.append(chunk.strip())
    return chunks

# Get context from surrounding chunks
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split("\n")[-3:]
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split("\n")[:2]

# 🔍 Translate a single chunk
def translate_chunk(paths: OutputPaths, chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(paths, chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# Add similarity calculation function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 🚀 Main function to translate all chunks
@check_file_exists(lambda paths, *_args, **_kwargs: paths.translation)
def translate_all(paths: OutputPaths, config: dict):
    output.info("Start translating all chunks")
    chunks = split_chunks_by_chars(
        paths,
        chunk_size=int(get_toml_value(config, "subtitle.translate.chunk_size", 600) or 600),
        max_i=int(get_toml_value(config, "subtitle.translate.max_lines_per_chunk", 10) or 10),
    )
    if not chunks:
        raise ValueError("No subtitle chunks found for translation")

    theme_prompt = None
    if paths.terminology.exists():
        with paths.terminology.open("r", encoding="utf-8") as file:
            theme_prompt = json.load(file).get("theme")

    max_workers = int(get_toml_value(config, "subtitle.max_workers", 4) or 4)
    progress = output.ProgressLine("Translating chunks").start()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(translate_chunk, paths, chunk, chunks, theme_prompt, i)
            futures.append(future)
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            progress.update(int(done_count * 100 / len(chunks)), f"Translating chunks {done_count}/{len(chunks)}")
    progress.finish("Chunk translation complete")

    results.sort(key=lambda x: x[0])

    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split("\n")
        src_text.extend(chunk_lines)

        chunk_text = "".join(chunk_lines).lower()
        matching_results = [(r, similar("".join(r[1].split("\n")).lower(), chunk_text)) for r in results]
        best_match = max(matching_results, key=lambda x: x[1])

        if best_match[1] < 0.9:
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            output.warn(f"Similar match found for chunk {i}, similarity: {best_match[1]:.3f}")

        trans_text.extend(best_match[0][2].split("\n"))

    df_text = pd.read_excel(paths.cleaned_chunks)
    df_text["text"] = df_text["text"].str.strip('"').str.strip()
    df_translate = pd.DataFrame({"Source": src_text, "Translation": trans_text})
    subtitle_output_configs = [("trans_subs_for_audio.srt", ["Translation"])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)

    min_trim_duration = float(get_toml_value(config, "subtitle.translate.min_trim_duration", 3.5) or 3.5)
    # df_time["Translation"] = df_time.apply(
    #     lambda x: check_len_then_trim(x["Translation"], x["duration"])
    #     if x["duration"] > min_trim_duration
    #     else x["Translation"],
    #     axis=1,
    # )

    df_time.to_excel(paths.translation, index=False)
    output.success(f"Translation completed and results saved to {paths.translation}")


# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str, config: dict) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)

    def valid_align(response_data):
        if "align" not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data["align"]) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}

    parsed = ask_gpt(align_prompt, resp_type="json", valid_def=valid_align, log_title="align_subs")
    align_data = parsed["align"]
    src_parts = src_part.split("\n")
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]

    language = get_toml_str(config, "transcribe.whisperx.language", default="auto") or "auto"
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)

    output.info("Aligned subtitle parts")
    output.info(f"SRC_LANG: {' || '.join(src_parts)}")
    output.info(f"TARGET_LANG: {' || '.join(tr_parts)}")

    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str], config: dict):
    max_sub_length = int(get_toml_value(config, "subtitle.max_length", 75) or 75)
    target_sub_multiplier = float(get_toml_value(config, "subtitle.target_multiplier", 1.0) or 1.0)
    remerged_tr_lines = tr_lines.copy()

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > max_sub_length or calc_len(tr) * target_sub_multiplier > max_sub_length:
            to_split.append(i)
            output.info(f"Line {i} needs split")
            output.info(f"Source Line: {src}")
            output.info(f"Target Line: {tr}")
    
    @except_handler("Error in split_align_subs")
    def process(i):
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src, config)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged

    max_workers = int(get_toml_value(config, "subtitle.max_workers", 4) or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process, to_split)

    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]

    return src_lines, tr_lines, remerged_tr_lines

@check_file_exists(lambda paths, *_args, **_kwargs: paths.remerged)
def split_for_sub_main(paths: OutputPaths, config: dict):
    output.info("Start splitting subtitles for subtitle display")

    df = pd.read_excel(paths.translation)
    src = df["Source"].tolist()
    trans = df["Translation"].tolist()

    max_sub_length = int(get_toml_value(config, "subtitle.max_length", 75) or 75)
    target_sub_multiplier = float(get_toml_value(config, "subtitle.target_multiplier", 1.0) or 1.0)

    split_src = src
    split_trans = trans
    remerged = trans
    for attempt in range(3):
        output.info(f"Split attempt {attempt + 1}")
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans, config)

        if all(len(str(src_line)) <= max_sub_length for src_line in split_src) and all(
            calc_len(str(tr_line)) * target_sub_multiplier <= max_sub_length for tr_line in split_trans
        ):
            break

        src, trans = split_src, split_trans

    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))

    pd.DataFrame({"Source": split_src, "Translation": split_trans}).to_excel(paths.split_sub, index=False)
    pd.DataFrame({"Source": src, "Translation": remerged}).to_excel(paths.remerged, index=False)
    output.success(f"Subtitle split results saved to {paths.split_sub}")
    output.success(f"Remerged subtitle results saved to {paths.remerged}")



# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ""
    cleaned = str(x).strip("。").strip("，")
    return autocorrect.format(cleaned)


def _all_paths_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)

@check_file_exists(lambda paths, *_args, **_kwargs: paths.trans_src_srt)
def align_timestamp_main(paths: OutputPaths):
    subtitle_outputs = [
        paths.src_srt,
        paths.trans_srt,
        paths.src_trans_srt,
        paths.trans_src_srt,
    ]

    df_text = pd.read_excel(paths.cleaned_chunks)
    df_text["text"] = df_text["text"].str.strip('"').str.strip()

    df_translate = pd.read_excel(paths.split_sub)
    df_translate["Translation"] = df_translate["Translation"].apply(clean_translation)
    align_timestamp(df_text, df_translate, SUBTITLE_OUTPUT_CONFIGS, paths.output_dir)
    output.success(f"Subtitle files generated in {paths.output_dir}")


def run(args: Namespace, config: dict) -> int:
    work_dir = get_work_dir(config) or "."
    paths = build_output_paths(work_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    try:
        split_by_spacy(paths)
        if get_toml_value(config, "subtitle.split_by_meaning", False):
            split_sentences_by_meaning(paths, config)
        if get_toml_value(config, "subtitle.generate_summary", False):
            get_summary(paths, config)
        
        translate_all(paths, config)
        if get_toml_value(config, "subtitle.split_for_sub", False):
            split_for_sub_main(paths, config)

        align_timestamp_main(paths)

    except Exception as e:
        output.error(str(e))
        return EXIT.RUNTIME_ERROR

    return EXIT.SUCCESS
