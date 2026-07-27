from dataclasses import dataclass

from subgen.subtitles import (
    format_srt_timestamp,
    format_vtt_timestamp,
    render_srt,
    render_vtt,
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def test_srt_and_vtt_timestamp_format() -> None:
    assert format_srt_timestamp(1.234) == "00:00:01,234"
    assert format_vtt_timestamp(1.234) == "00:00:01.234"


def test_timestamp_crosses_hour_and_rounds_millisecond_boundary() -> None:
    assert format_srt_timestamp(3661.9996) == "01:01:02,000"
    assert format_vtt_timestamp(3661.9996) == "01:01:02.000"


def test_render_srt_skips_empty_segments_and_keeps_numbering_continuous() -> None:
    segments = [
        Segment(0, 1, "第一句。"),
        Segment(1, 2, "   "),
        Segment(2, 3, "第二句！"),
    ]

    assert render_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "第一句。\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "第二句！\n"
    )


def test_render_vtt_has_header_and_skips_empty_segments() -> None:
    segments = [
        Segment(0, 1.5, ""),
        Segment(1.5, 2.25, "网页字幕"),
    ]

    assert render_vtt(segments) == (
        "WEBVTT\n"
        "\n"
        "00:00:01.500 --> 00:00:02.250\n"
        "网页字幕\n"
    )
