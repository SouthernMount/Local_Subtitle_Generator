"""术语表（人名/专有名词）支持。

人名是整条流水线上最容易"前后不一致"的部分：Whisper 可能把同一个名字听成
两种写法，翻译模型又会自己音译（``ヒカル`` → ``希卡尔``）。术语表的作用不是
"查字典"，而是给出一个**唯一指定**的中文写法，并顺手把人名送进 Whisper 的
hotwords、写进翻译提示词。

格式很宽松，一行（或用逗号/分号分隔）一条，用 ``=`` / ``→`` / ``:`` 连接::

    # 角色名
    ヒカル=光
    みえ → 美惠
    タカシ: 隆

本模块只依赖标准库，方便离线使用和单元测试。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .postprocess import normalize_japanese

# 词条之间的分隔：换行、中英文逗号、分号、顿号。
_ENTRY_SPLIT = re.compile(r"[\n\r;；,，、]+")
# 词条内部 "原文=译文" 的分隔符；先匹配多字符的箭头，避免被 "-" 抢先。
_PAIR_SPLIT = re.compile(r"\s*(?:=>|->|→|＝|=|:)\s*")
# hotwords 已经是一个自由文本提示，这里只按常见分隔符去重。
_HOTWORD_SPLIT = re.compile(r"[\n\r;；,，、/\s]+")


@dataclass(frozen=True)
class GlossaryEntry:
    """一条术语：``source`` 是素材里的写法，``target`` 是要求输出的中文写法。"""

    source: str
    target: str


def parse_glossary(text: str | None) -> list[GlossaryEntry]:
    """把用户输入的文本解析成术语表；无法解析的片段直接忽略。

    重复的 ``source`` 以最后一次为准，顺序保持首次出现的顺序。
    """
    if not text:
        return []
    ordered: dict[str, str] = {}
    for raw_entry in _ENTRY_SPLIT.split(text):
        entry = raw_entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = _PAIR_SPLIT.split(entry, maxsplit=1)
        if len(parts) != 2:
            continue
        source = normalize_japanese(parts[0])
        target = normalize_japanese(parts[1])
        if not source or not target:
            continue
        ordered.pop(source, None)
        ordered[source] = target
    return [GlossaryEntry(source=source, target=target) for source, target in ordered.items()]


def load_glossary_file(path: Path) -> list[GlossaryEntry]:
    """读取术语表文件（UTF-8，允许带 BOM）。"""
    text = path.expanduser().read_text(encoding="utf-8-sig")
    return parse_glossary(text)


def collect_glossary(inline: str | None, path: Path | None) -> list[GlossaryEntry]:
    """合并术语表文件与内联写法；同一个原词以内联写法为准。"""
    entries: list[GlossaryEntry] = []
    if path is not None:
        resolved = path.expanduser()
        if not resolved.is_file():
            raise ValueError(f"Glossary file does not exist: {resolved}")
        entries.extend(load_glossary_file(resolved))
    entries.extend(parse_glossary(inline))
    merged: dict[str, GlossaryEntry] = {}
    for entry in entries:
        merged[entry.source] = entry
    return list(merged.values())


def build_hotwords(
    explicit: str | None,
    entries: Sequence[GlossaryEntry] = (),
    *,
    max_chars: int = 200,
) -> str | None:
    """把 ``--hotwords`` 和术语表的原词合并成一个提示串。

    Whisper 的 hotwords 本质是一段引导文本，太长会挤占上下文并诱发复读，
    因此这里做去重并限制总长度（默认 200 字符）。
    """
    parts: list[str] = []
    seen: set[str] = set()
    candidates = list(_HOTWORD_SPLIT.split(explicit or "")) + [
        entry.source for entry in entries
    ]
    for candidate in candidates:
        token = normalize_japanese(candidate)
        if not token or token in seen:
            continue
        seen.add(token)
        parts.append(token)
    if not parts:
        return None
    hotwords = ""
    for token in parts:
        candidate = f"{hotwords}、{token}" if hotwords else token
        if len(candidate) > max_chars:
            break
        hotwords = candidate
    return hotwords or None


def parse_renames(text: str | None) -> list[tuple[str, str]]:
    """解析"旧中文=新中文"的译名替换规则，供只看中文的用户使用。

    与术语表共用分隔规则，但两侧都是中文：用户读到字幕里的"希卡尔"，
    就能直接把它改成"光"，不需要知道日语原词。
    """
    if not text:
        return []
    ordered: dict[str, str] = {}
    for raw_entry in _ENTRY_SPLIT.split(text):
        entry = raw_entry.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = _PAIR_SPLIT.split(entry, maxsplit=1)
        if len(parts) != 2:
            continue
        old = normalize_japanese(parts[0])
        new = normalize_japanese(parts[1])
        if not old or not new or old == new:
            continue
        ordered.pop(old, None)
        ordered[old] = new
    return list(ordered.items())


def describe_glossary(entries: Sequence[GlossaryEntry], *, limit: int = 5) -> str:
    """进度提示用的一句话摘要。"""
    shown = "、".join(f"{entry.source}→{entry.target}" for entry in entries[:limit])
    suffix = " 等" if len(entries) > limit else ""
    return f"术语表 {len(entries)} 条（{shown}{suffix}）"
