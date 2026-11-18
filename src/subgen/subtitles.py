from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol


class SubtitleLike(Protocol):
    start: float
    end: float
    text: str


def format_srt_timestamp(seconds: float) -> str:
    return _format_timestamp(seconds, millisecond_separator=",")


def format_vtt_timestamp(seconds: float) -> str:
    return _format_timestamp(seconds, millisecond_separator=".")


def wrap_cjk_text(text: str, *, max_chars: int = 20, max_lines: int = 2) -> str:
    """按显示宽度折行：优先在标点处断开，最多 ``max_lines`` 行。

    中文/日文没有词边界，按字符数折行即可；超出 ``max_chars * max_lines``
    的部分继续留在最后一行，避免丢字。
    """
    stripped = text.strip()
    if not stripped or max_chars <= 0 or len(stripped) <= max_chars:
        return stripped

    lines: list[str] = []
    remainder = stripped
    while len(remainder) > max_chars and len(lines) < max_lines - 1:
        window = remainder[: max_chars + 1]
        cut = max((window.rfind(mark) + 1 for mark in "。！？!?、，,；;…"), default=-1)
        if cut < max_chars // 2:
            cut = max_chars
        lines.append(remainder[:cut].strip())
        remainder = remainder[cut:].lstrip()
    if remainder:
        lines.append(remainder)
    return "\n".join(line for line in lines if line)


def render_srt(
    segments: Iterable[SubtitleLike],
    *,
    wrap_chars: int | None = None,
    bilingual: bool = False,
) -> str:
    blocks: list[str] = []
    index = 1

    for segment in _non_empty_segments(segments):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}",
                    _render_text(segment, wrap_chars=wrap_chars, bilingual=bilingual),
                ]
            )
        )
        index += 1

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(
    segments: Iterable[SubtitleLike],
    *,
    wrap_chars: int | None = None,
    bilingual: bool = False,
) -> str:
    blocks = ["WEBVTT"]

    for segment in _non_empty_segments(segments):
        blocks.append(
            "\n".join(
                [
                    f"{format_vtt_timestamp(segment.start)} --> {format_vtt_timestamp(segment.end)}",
                    _render_text(segment, wrap_chars=wrap_chars, bilingual=bilingual),
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def write_srt(
    path: Path,
    segments: Iterable[SubtitleLike],
    *,
    wrap_chars: int | None = None,
    bilingual: bool = False,
) -> None:
    path.write_text(
        render_srt(segments, wrap_chars=wrap_chars, bilingual=bilingual),
        encoding="utf-8",
    )


def write_vtt(
    path: Path,
    segments: Iterable[SubtitleLike],
    *,
    wrap_chars: int | None = None,
    bilingual: bool = False,
) -> None:
    path.write_text(
        render_vtt(segments, wrap_chars=wrap_chars, bilingual=bilingual),
        encoding="utf-8",
    )


def _render_text(
    segment: SubtitleLike,
    *,
    wrap_chars: int | None,
    bilingual: bool,
) -> str:
    text = segment.text.strip()
    source = getattr(segment, "source_text", None)
    if bilingual and isinstance(source, str) and source.strip() and source.strip() != text:
        lines = [source.strip(), text]
    else:
        lines = [text]
    if wrap_chars:
        lines = [wrap_cjk_text(line, max_chars=wrap_chars) for line in lines]
    return "\n".join(line for line in lines if line)


def _format_timestamp(seconds: float, *, millisecond_separator: str) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    milliseconds = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return (
        f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
        f"{millisecond_separator}{milliseconds:03d}"
    )


def _non_empty_segments(segments: Iterable[SubtitleLike]) -> Iterable[SubtitleLike]:
    for segment in segments:
        if segment.text.strip():
            yield segment
