import string
import warnings

from my_video.cli import output
from my_video.core.spacy_utils.load_nlp_model import init_nlp, resolve_spacy_language
from my_video.core.utils import get_joiner
from my_video.core.utils.models import OutputPaths

warnings.filterwarnings("ignore", category=FutureWarning)


def split_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)

    dp = [float("inf")] * (n + 1)
    dp[0] = 0
    prev = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(max(0, i - 100), i):
            if i - j >= 30:
                token = doc[i - 1]
                if j == 0 or (token.is_sent_end or token.pos_ in ["VERB", "AUX"] or token.dep_ == "ROOT"):
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

    sentences = []
    i = n
    language = resolve_spacy_language()
    joiner = get_joiner(language)
    while i > 0:
        j = prev[i]
        sentences.append(joiner.join(tokens[j:i]).strip())
        i = j

    return sentences[::-1]


def split_extremely_long_sentence(doc):
    tokens = [token.text for token in doc]
    n = len(tokens)

    num_parts = (n + 59) // 60
    part_length = n // num_parts

    sentences = []
    language = resolve_spacy_language()
    joiner = get_joiner(language)
    for i in range(num_parts):
        start = i * part_length
        end = start + part_length if i < num_parts - 1 else n
        sentence = joiner.join(tokens[start:end])
        sentences.append(sentence)
    
    return sentences


def split_long_by_root_main(nlp, paths: OutputPaths):
    with paths.split_by_connector.open("r", encoding="utf-8") as input_file:
        sentences = input_file.readlines()

    all_split_sentences = []
    for sentence in sentences:
        doc = nlp(sentence.strip())
        if len(doc) > 60:
            split_sentences = split_long_sentence(doc)
            if any(len(nlp(sent)) > 60 for sent in split_sentences):
                split_sentences = [
                    subsent for sent in split_sentences for subsent in split_extremely_long_sentence(nlp(sent))
                ]
            all_split_sentences.extend(split_sentences)
            output.info(f"Splitting long sentences by root: {sentence[:30]}...")
        else:
            all_split_sentences.append(sentence.strip())

    punctuation = string.punctuation + "'" + '"'

    with paths.split_by_nlp.open("w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(all_split_sentences):
            stripped_sentence = sentence.strip()
            if not stripped_sentence or all(char in punctuation for char in stripped_sentence):
                output.warn(f"Empty or punctuation-only line detected at index {i}")
                if i > 0:
                    all_split_sentences[i - 1] += sentence
                continue
            output_file.write(sentence + "\n")

    # os.remove(paths.split_by_connector)

    output.success(f"Long sentences split by root saved to {paths.split_by_nlp}")

if __name__ == "__main__":
    nlp = init_nlp()
    from my_video.core.utils.models import build_output_paths

    split_long_by_root_main(nlp, build_output_paths("."))
