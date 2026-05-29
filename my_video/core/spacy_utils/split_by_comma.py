import itertools
import warnings

from my_video.cli import output
from my_video.core.spacy_utils.load_nlp_model import init_nlp
from my_video.core.utils.models import OutputPaths

warnings.filterwarnings("ignore", category=FutureWarning)


def is_valid_phrase(phrase):
    has_subject = any(token.dep_ in ["nsubj", "nsubjpass"] or token.pos_ == "PRON" for token in phrase)
    has_verb = any(token.pos_ in ["VERB", "AUX"] for token in phrase)
    return has_subject and has_verb


def analyze_comma(start, doc, token):
    left_phrase = doc[max(start, token.i - 9) : token.i]
    right_phrase = doc[token.i + 1 : min(len(doc), token.i + 10)]

    suitable_for_splitting = is_valid_phrase(right_phrase)

    left_words = [t for t in left_phrase if not t.is_punct]
    right_words = list(itertools.takewhile(lambda t: not t.is_punct, right_phrase))

    if len(left_words) <= 3 or len(right_words) <= 3:
        suitable_for_splitting = False

    return suitable_for_splitting


def split_by_comma(text, nlp):
    doc = nlp(text)
    sentences = []
    start = 0
    
    for i, token in enumerate(doc):
        if token.text == "," or token.text == "，":
            suitable_for_splitting = analyze_comma(start, doc, token)
            if suitable_for_splitting:
                sentences.append(doc[start:token.i].text.strip())
                output.info(f"Split at comma near: {doc[start:token.i][-4:]},| {doc[token.i + 1:][:4]}")
                start = token.i + 1

    sentences.append(doc[start:].text.strip())
    return sentences


def split_by_comma_main(nlp, paths: OutputPaths):
    with paths.split_by_mark.open("r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        split_sentences = split_by_comma(sentence.strip(), nlp)
        all_split_sentences.extend(split_sentences)

    with paths.split_by_comma.open("w", encoding="utf-8") as output_file:
        for sentence in all_split_sentences:
            output_file.write(sentence + "\n")

    # os.remove(paths.split_by_mark)

    output.success(f"Sentences split by commas saved to {paths.split_by_comma}")

if __name__ == "__main__":
    nlp = init_nlp()
    from my_video.core.utils.models import build_output_paths

    split_by_comma_main(nlp, build_output_paths("."))
