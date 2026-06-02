from pathlib import Path

from my_video.core.asr.asr_data import SubtitleLine


def format_srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(max(0, milliseconds), 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def write_subtitle_lines_to_srt(
    lines: list[SubtitleLine],
    path: Path,
    *,
    use_translation: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for index, line in enumerate(lines, start=1):
        text = line.translate_text if use_translation else line.text
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(line.start_time)} --> {format_srt_timestamp(line.end_time)}",
                    text,
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
