"""subtitle command — lightweight placeholder for subtitle processing."""

from argparse import Namespace
from pathlib import Path

from my_video.cli import output
from my_video.cli import exit_codes as EXIT
from my_video.cli.config import get_work_dir
from my_video.core.spacy_utils import (
    init_nlp,
    split_by_comma_main,
    split_by_mark,
    split_long_by_root_main,
    split_sentences_main,
)
from my_video.core.utils import check_file_exists
from my_video.core.utils.models import OutputPaths, build_output_paths


@check_file_exists(lambda paths, *_args, **_kwargs: paths.split_by_nlp)
def split_by_spacy(paths: OutputPaths):
    nlp = init_nlp()
    split_by_mark(nlp, paths)
    split_by_comma_main(nlp, paths)
    split_sentences_main(nlp, paths)
    split_long_by_root_main(nlp, paths)


def run(args: Namespace, config: dict) -> int:
    # input_path = Path(args.input)
    # if not input_path.exists():
    #     output.error(f"Input file not found: {input_path}")
    #     return EXIT.FILE_NOT_FOUND

    work_dir = get_work_dir(config) or "."
    paths = build_output_paths(work_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    quiet = getattr(args, "quiet", False)

    try:
        split_by_spacy(paths)
    except Exception as e:
        output.error(str(e))
        return EXIT.RUNTIME_ERROR

    if quiet:
        print(paths.split_by_nlp)
    return EXIT.SUCCESS
