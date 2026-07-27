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


def render_srt(segments: Iterable[SubtitleLike]) -> str:
    blocks: list[str] = []
    index = 1

    for segment in _non_empty_segments(segments):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}",
                    segment.text.strip(),
                ]
            )
        )
        index += 1

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: Iterable[SubtitleLike]) -> str:
    blocks = ["WEBVTT"]

    for segment in _non_empty_segments(segments):
        blocks.append(
            "\n".join(
                [
                    f"{format_vtt_timestamp(segment.start)} --> {format_vtt_timestamp(segment.end)}",
                    segment.text.strip(),
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def write_srt(path: Path, segments: Iterable[SubtitleLike]) -> None:
    path.write_text(render_srt(segments), encoding="utf-8")


def write_vtt(path: Path, segments: Iterable[SubtitleLike]) -> None:
    path.write_text(render_vtt(segments), encoding="utf-8")


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
