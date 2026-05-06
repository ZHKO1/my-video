NO_SPACE_LANGUAGES = {"zh", "ja", "ko", "th"}


def get_joiner(language: str) -> str:
    return "" if language.lower() in NO_SPACE_LANGUAGES else " "
