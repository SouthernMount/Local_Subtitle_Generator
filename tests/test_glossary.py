from pathlib import Path

import pytest

from subgen.glossary import (
    GlossaryEntry,
    build_hotwords,
    collect_glossary,
    describe_glossary,
    parse_glossary,
    parse_renames,
)


def test_parse_glossary_accepts_common_separators() -> None:
    entries = parse_glossary("ヒカル=光\nみえ → 美惠; タカシ: 隆, # 注释\n")

    assert entries == [
        GlossaryEntry("ヒカル", "光"),
        GlossaryEntry("みえ", "美惠"),
        GlossaryEntry("タカシ", "隆"),
    ]


def test_parse_glossary_ignores_broken_lines_and_keeps_last_duplicate() -> None:
    entries = parse_glossary("ヒカル\nヒカル=光\nヒカル=小光\n\n=")

    assert entries == [GlossaryEntry("ヒカル", "小光")]


def test_parse_glossary_of_empty_text_is_empty() -> None:
    assert parse_glossary(None) == []
    assert parse_glossary("  ") == []


def test_collect_glossary_prefers_inline_over_file(tmp_path: Path) -> None:
    path = tmp_path / "glossary.tsv"
    path.write_text("ヒカル=小光\nみえ=美惠\n", encoding="utf-8")

    entries = collect_glossary("ヒカル=光", path)

    assert entries == [GlossaryEntry("ヒカル", "光"), GlossaryEntry("みえ", "美惠")]


def test_collect_glossary_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Glossary file does not exist"):
        collect_glossary(None, tmp_path / "missing.tsv")


def test_build_hotwords_merges_and_dedupes() -> None:
    hotwords = build_hotwords(
        "ヒカル, タカシ",
        [GlossaryEntry("ヒカル", "光"), GlossaryEntry("みえ", "美惠")],
    )

    assert hotwords == "ヒカル、タカシ、みえ"


def test_build_hotwords_respects_length_cap() -> None:
    entries = [GlossaryEntry(f"なまえ{index:02d}", "名字") for index in range(30)]

    hotwords = build_hotwords(None, entries, max_chars=20)

    assert hotwords is not None
    assert len(hotwords) <= 20
    assert hotwords.startswith("なまえ00")


def test_build_hotwords_without_input_is_none() -> None:
    assert build_hotwords(None, []) is None


def test_parse_renames_accepts_chinese_pairs() -> None:
    assert parse_renames("希卡尔=光, 美惠 → 美由纪") == [
        ("希卡尔", "光"),
        ("美惠", "美由纪"),
    ]


def test_parse_renames_ignores_identical_and_incomplete_pairs() -> None:
    assert parse_renames("光=光, = 光, 美惠 =, 只有左边") == []


def test_describe_glossary_truncates_long_lists() -> None:
    entries = [GlossaryEntry(f"a{index}", f"甲{index}") for index in range(7)]

    assert "等" in describe_glossary(entries)
    assert "a5" not in describe_glossary(entries)
