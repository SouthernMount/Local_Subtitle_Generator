from dataclasses import dataclass

from subgen.postprocess import (
    apply_renames,
    build_quality_report,
    dedupe_consecutive,
    drop_hallucinations,
    enforce_timing,
    estimate_cue_duration,
    finalize_subtitles,
    is_hallucination,
    is_interjection,
    kana_ratio,
    merge_short_segments,
    normalize_chinese,
    normalize_japanese,
    repair_timestamps,
    split_long_segments,
    translate_interjection,
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    source_text: str | None = None


def test_normalize_chinese_fixes_punctuation_and_leading_comma() -> None:
    assert normalize_chinese(",是的,是的.") == "是的，是的。"
    assert normalize_chinese("这是一个 测试 用的 句子") == "这是一个测试用的句子"
    assert normalize_chinese("你好。") == "你好。"


def test_normalize_chinese_strips_trailing_comma_from_split() -> None:
    assert normalize_chinese("我们先吃饭，") == "我们先吃饭"


def test_normalize_japanese_keeps_full_width_punctuation() -> None:
    assert normalize_japanese("　こんにちは！　") == "こんにちは！"


def test_kana_ratio_distinguishes_kana_and_kanji() -> None:
    assert kana_ratio("こんにちは") == 1.0
    assert kana_ratio("東京駅") == 0.0
    assert 0.0 < kana_ratio("東京で買い物") < 1.0


def test_is_hallucination_matches_known_outro_but_keeps_real_thanks() -> None:
    assert is_hallucination("ご視聴ありがとうございました。") is True
    assert is_hallucination("チャンネル登録お願いします") is True
    assert is_hallucination("ありがとうございました。") is False
    assert is_hallucination("……") is True


def test_drop_hallucinations_removes_outro_and_latin_noise() -> None:
    segments = [
        Segment(0, 1, "今日はいい天気ですね。"),
        Segment(1, 2, "ご視聴ありがとうございました。"),
        Segment(2, 3, "Thank you for watching."),
    ]

    kept = drop_hallucinations(segments, language="ja")

    assert [segment.text for segment in kept] == ["今日はいい天気ですね。"]


def test_dedupe_consecutive_keeps_two_repeats() -> None:
    segments = [Segment(index, index + 1, "また同じ台詞。") for index in range(4)]

    assert len(dedupe_consecutive(segments)) == 2


def test_merge_short_segments_joins_fragment() -> None:
    segments = [Segment(0.0, 0.4, "えっと、"), Segment(0.6, 2.0, "ちょっと待ってください。")]

    merged = merge_short_segments(segments)

    assert len(merged) == 1
    assert merged[0].text == "えっと、ちょっと待ってください。"
    assert merged[0].end == 2.0


def test_merge_short_segments_keeps_long_gap() -> None:
    segments = [Segment(0.0, 0.4, "はい。"), Segment(5.0, 7.0, "次の話題です。")]

    assert len(merge_short_segments(segments)) == 2


def test_split_long_segments_respects_max_chars() -> None:
    long_text = "これはとても長い字幕で、プレイヤーの中で画面からはみ出してしまいます。"

    pieces = split_long_segments([Segment(0.0, 6.0, long_text)], max_chars=20)

    assert len(pieces) > 1
    assert all(len(piece.text) <= 20 for piece in pieces)
    assert "".join(piece.text for piece in pieces) == long_text
    assert pieces[0].start == 0.0
    assert pieces[-1].end == 6.0


def test_enforce_timing_fills_min_duration_and_removes_overlap() -> None:
    segments = [Segment(0.0, 0.1, "一。"), Segment(0.2, 0.5, "二。")]

    fixed = enforce_timing(segments, min_duration=0.8)

    assert fixed[0].end - fixed[0].start >= 0.2
    assert fixed[0].end <= fixed[1].start
    assert fixed[1].end - fixed[1].start >= 0.2


def test_finalize_subtitles_runs_full_pipeline() -> None:
    segments = [
        Segment(0, 1, "ご視聴ありがとうございました。"),
        Segment(1, 1.2, "今日は"),
        Segment(1.2, 1.4, "いい天気ですね。"),
    ]

    final = finalize_subtitles(segments, language="ja", max_chars=20)

    assert [segment.text for segment in final] == ["今日はいい天気ですね。"]
    assert final[0].start == 1


def test_build_quality_report_flags_language_drift() -> None:
    rows = build_quality_report(
        [Segment(0, 1, "这是中文。", source_text="这是一个没有假名的中文句子")],
        language="ja",
    )

    assert rows[0]["flags"] == ["low_kana"]
    assert rows[0]["text"] == "这是中文。"
    assert rows[0]["source"] == "这是一个没有假名的中文句子"


def test_estimate_cue_duration_is_bounded() -> None:
    assert estimate_cue_duration("あ") == 1.0
    assert estimate_cue_duration("あ" * 60) == 8.0
    assert 1.0 < estimate_cue_duration("あ" * 12) < 8.0


def test_repair_timestamps_clamps_absurd_span() -> None:
    segments = [Segment(1580.0, 2818.0, "その時、ロゲーと")]

    fixed = repair_timestamps(segments)

    assert fixed[0].start == 1580.0
    assert fixed[0].end - fixed[0].start <= 2.0


def test_repair_timestamps_keeps_normal_segment() -> None:
    segments = [Segment(0.0, 3.0, "今日はいい天気ですね。")]

    assert repair_timestamps(segments) == segments


def test_repair_timestamps_never_overlaps_next_cue() -> None:
    segments = [Segment(10.0, 900.0, "はい。"), Segment(12.0, 13.0, "次の話題です。")]

    fixed = repair_timestamps(segments)

    assert fixed[0].end <= 12.0
    assert fixed[1] == segments[1]


def test_translate_interjection_uses_fixed_rendering() -> None:
    assert translate_interjection("はい。") == "嗯"
    assert translate_interjection("「うん」") == "嗯"
    assert translate_interjection("はい、はい") == "嗯，嗯"
    assert translate_interjection("ん？") == "嗯？"


def test_translate_interjection_returns_none_for_real_sentences() -> None:
    assert translate_interjection("はい、わかりました。") is None
    assert translate_interjection("今日は寒いですね") is None


def test_is_interjection_recognizes_both_languages() -> None:
    assert is_interjection("はい")
    assert is_interjection("嗯")
    assert is_interjection("そう、そう")

    assert not is_interjection("今日は寒いですね")


def test_merge_short_segments_keeps_reply_separate() -> None:
    segments = [Segment(0.0, 0.6, "我知道了。"), Segment(0.6, 1.0, "嗯")]

    merged = merge_short_segments(segments)

    assert [segment.text for segment in merged] == ["我知道了。", "嗯"]


def test_merge_short_segments_still_merges_normal_fragments() -> None:
    segments = [Segment(0.0, 0.6, "今日は"), Segment(0.6, 1.0, "寒いですね")]

    merged = merge_short_segments(segments)

    assert [segment.text for segment in merged] == ["今日は寒いですね"]


def test_enforce_timing_gives_short_replies_more_time() -> None:
    fixed = enforce_timing([Segment(0.0, 0.2, "嗯")])

    assert fixed[0].end - fixed[0].start >= 1.2


def test_apply_renames_replaces_longest_match_first() -> None:
    renames = [("希卡", "光"), ("希卡尔", "小光")]

    assert apply_renames("希卡尔和希卡", renames) == "小光和光"


def test_apply_renames_ignores_empty_and_identical_pairs() -> None:
    assert apply_renames("你好", [("", "x"), ("你好", "你好")]) == "你好"
