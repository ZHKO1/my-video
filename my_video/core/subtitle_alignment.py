import re
from pathlib import Path


from my_video.cli import output


def convert_to_srt_format(start_time, end_time):
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"


def remove_punctuation(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []

    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)

    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))

    output.warn("Difference positions:")
    output.warn(f"Expected sentence: {str1}")
    output.warn(f"Actual match: {str2}")
    output.warn(
        "Position markers: "
        + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2))))
    )
    output.warn(f"Difference indices: {diff_positions}")


def get_sentence_timestamps(df_words, df_sentences):
    time_stamp_list = []
    full_words_str = ""
    position_to_word_idx = {}

    for idx, word in enumerate(df_words["text"]):
        clean_word = remove_punctuation(str(word).lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx

    current_pos = 0
    for idx, sentence in df_sentences["Source"].items():
        clean_sentence = remove_punctuation(str(sentence).lower()).replace(" ", "")
        sentence_len = len(clean_sentence)

        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos : current_pos + sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                time_stamp_list.append(
                    (
                        float(df_words["start"][start_word_idx]),
                        float(df_words["end"][end_word_idx]),
                    )
                )
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1

        if not match_found:
            output.warn(f"No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, full_words_str[current_pos : current_pos + len(clean_sentence)])
            output.warn(f"Original sentence: {df_sentences['Source'][idx]}")
            raise ValueError(f"No match found for sentence: {sentence}")

    return time_stamp_list


def align_timestamp(
    df_text,
    df_translate,
    subtitle_output_configs: list[tuple[str, list[str]]],
    output_dir: str | Path | None,
    for_display: bool = True,
):
    df_trans_time = df_translate.copy()
    time_stamp_list = get_sentence_timestamps(df_text, df_translate)
    df_trans_time["timestamp"] = time_stamp_list
    df_trans_time["duration"] = df_trans_time["timestamp"].apply(lambda x: x[1] - x[0])

    for i in range(len(df_trans_time) - 1):
        delta_time = df_trans_time.loc[i + 1, "timestamp"][0] - df_trans_time.loc[i, "timestamp"][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, "timestamp"] = (
                df_trans_time.loc[i, "timestamp"][0],
                df_trans_time.loc[i + 1, "timestamp"][0],
            )

    df_trans_time["timestamp"] = df_trans_time["timestamp"].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    if for_display:
        df_trans_time["Translation"] = df_trans_time["Translation"].apply(
            lambda x: re.sub(r"[，。]", " ", str(x)).strip()
        )

    def generate_subtitle_string(df, columns):
        return "".join(
            [
                (
                    f"{i + 1}\n{row['timestamp']}\n{row[columns[0]].strip()}\n"
                    f"{row[columns[1]].strip() if len(columns) > 1 else ''}\n\n"
                )
                for i, row in df.iterrows()
            ]
        ).strip()

    if output_dir:
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            (base / filename).write_text(subtitle_str, encoding="utf-8")

    return df_trans_time
