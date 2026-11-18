from dataclasses import dataclass

from subgen.subtitles import (
    format_srt_timestamp,
    format_vtt_timestamp,
    render_srt,
    render_vtt,
    wrap_cjk_text,
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class BilingualSegment:
    start: float
    end: float
    text: str
    source_text: str | None = None


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


def test_wrap_cjk_text_splits_evenly_without_punctuation() -> None:
    assert wrap_cjk_text("あ" * 20, max_chars=10) == "あ" * 10 + "\n" + "あ" * 10


def test_wrap_cjk_text_prefers_punctuation_boundary() -> None:
    wrapped = wrap_cjk_text("これはテストです、次の文です。", max_chars=12)

    assert wrapped.split("\n")[0].endswith("、")


def test_render_srt_wraps_when_wrap_chars_given() -> None:
    rendered = render_srt([Segment(0, 1, "あ" * 20)], wrap_chars=10)

    assert "あ" * 10 + "\n" + "あ" * 10 in rendered


def test_render_srt_bilingual_puts_source_before_translation() -> None:
    rendered = render_srt(
        [BilingualSegment(0, 1, "你好。", source_text="こんにちは。")],
        bilingual=True,
    )

    assert rendered.endswith("こんにちは。\n你好。\n")


def test_render_srt_bilingual_falls_back_when_source_missing() -> None:
    rendered = render_srt([Segment(0, 1, "你好。")], bilingual=True)

    assert rendered.endswith("\n你好。\n")
