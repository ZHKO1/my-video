import os
import warnings

import pandas as pd

from my_video.cli import output
from my_video.core.spacy_utils.load_nlp_model import init_nlp, resolve_spacy_language
from my_video.core.utils import get_joiner
from my_video.core.utils.models import OutputPaths

warnings.filterwarnings("ignore", category=FutureWarning)


def split_by_mark(nlp, paths: OutputPaths):
    language = resolve_spacy_language()
    joiner = get_joiner(language)
    output.info(f"Using {language} language joiner: {joiner!r}")
    chunks = pd.read_excel(paths.cleaned_chunks)
    chunks.text = chunks.text.apply(lambda x: x.strip('"').strip(""))

    input_text = joiner.join(chunks.text.to_list())

    doc = nlp(input_text)
    assert doc.has_annotation("SENT_START")

    sentences_by_mark = []
    current_sentence = []

    for sent in doc.sents:
        text = sent.text.strip()
        if current_sentence and (
            text.startswith("-")
            or text.startswith("...")
            or current_sentence[-1].endswith("-")
            or current_sentence[-1].endswith("...")
        ):
            current_sentence.append(text)
        else:
            if current_sentence:
                sentences_by_mark.append(' '.join(current_sentence))
                current_sentence = []
            current_sentence.append(text)
    
    if current_sentence:
        sentences_by_mark.append(" ".join(current_sentence))

    with paths.split_by_mark.open("w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(sentences_by_mark):
            if i > 0 and sentence.strip() in [",", ".", "，", "。", "？", "！"]:
                output_file.seek(output_file.tell() - 1, os.SEEK_SET)
                output_file.write(sentence)
            else:
                output_file.write(sentence + "\n")

    output.success(f"Sentences split by punctuation marks saved to {paths.split_by_mark}")

if __name__ == "__main__":
    nlp = init_nlp()
    from my_video.core.utils.models import build_output_paths

    split_by_mark(nlp, build_output_paths("."))
