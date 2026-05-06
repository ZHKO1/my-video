from pathlib import Path
from tempfile import TemporaryDirectory

from my_video.core.spacy_utils import init_nlp, split_long_by_root_main, split_sentences_main
from my_video.core.spacy_utils.split_by_comma import split_by_comma
from my_video.core.utils.models import build_output_paths


DEFAULT_SENTENCE = (
    "the team known as Fail Overflow, a hacking group known for reverse engineering "
    "of security models found in consumer electronics, performed a presentation at "
    "the 27th Chaos Communications Congress technical conference of their "
    "accomplishments with the PlayStation 3."
)


def _read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    sentence = DEFAULT_SENTENCE
    print("Input sentence:")
    print(sentence)
    print()

    nlp = init_nlp()

    comma_result = split_by_comma(sentence, nlp)
    print("split_by_comma result:")
    for index, item in enumerate(comma_result, start=1):
        print(f"{index}. {item}")
    print()

    with TemporaryDirectory(prefix="my-video-spacy-test-") as tmp_dir:
        paths = build_output_paths(tmp_dir)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.log_dir.mkdir(parents=True, exist_ok=True)

        paths.split_by_comma.write_text("\n".join(comma_result) + "\n", encoding="utf-8")
        split_sentences_main(nlp, paths)
        connector_result = _read_lines(paths.split_by_connector)

        print("split_sentences_main result:")
        for index, item in enumerate(connector_result, start=1):
            print(f"{index}. {item}")
        print()

        split_long_by_root_main(nlp, paths)
        long_root_result = _read_lines(paths.split_by_nlp)

        print("split_long_by_root_main result:")
        for index, item in enumerate(long_root_result, start=1):
            print(f"{index}. {item}")
        print()


if __name__ == "__main__":
    main()
