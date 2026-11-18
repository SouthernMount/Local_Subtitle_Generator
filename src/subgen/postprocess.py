"""识别与翻译结果的后处理：幻觉过滤、去重、合并、拆分、标点规范化。

Whisper 在日语视频上最典型的失败形态不是"字认错"，而是：
- 静音/音乐段吐出 ``ご視聴ありがとうございました`` 这类幻觉句；
- 长视频里同一句重复十几遍；
- 单条字幕过长（播放器里溢出）或过短（一闪而过）；
- 中文译文里混着半角逗号、句首逗号。
本模块按固定顺序把这些问题清掉，并且只依赖标准库。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Iterable, Protocol


class SubtitleSegment(Protocol):
    start: float
    end: float
    text: str
    source_text: str | None


# ---------------------------------------------------------------------------
# 常见的 Whisper 幻觉句（日语/英语/中文素材上高频出现）
# ---------------------------------------------------------------------------
_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"^ご視聴ありがとうござ(い|いました)[。、！!]?$",
        r"^ご視聴いただきありがとうござ(い|いました)[。、！!]?$",
        r"^チャンネル登録.*$",
        r"^登録お願いします[。！!]?$",
        r"^最後までご視聴.*$",
        r"^おやすみなさい[。、！!]?$",
        r"^字幕(by|は|提供).*$",
        r"^(thank you|thanks) for watching.*$",
        r"^please subscribe.*$",
        r"^subscribe to my channel.*$",
        r"^(谢谢|感谢)(大家)?(的)?(观看|收看|收看本视频)[。！!]?$",
        r"^(请)?(点赞|订阅|关注).*$",
        r"^字幕由.*提供.*$",
    )
)

# 这些符号单独成句时没有信息量。
_MEANINGLESS = {"", "。", "、", "，", "！", "？", "…", ".", ",", "!", "?", " ", "-", "—", "♪", "・"}

# ---------------------------------------------------------------------------
# 短句 / 语气词
# ---------------------------------------------------------------------------
# 单独成句的应答词不该交给翻译模型：大模型会把「はい」扩写成「好的，我明白了」
# 这类完整句，语气和长度都不对。这里给常用应答词一个固定译法，命中就跳过翻译。
INTERJECTION_TRANSLATIONS: dict[str, str] = {
    "はい": "嗯",
    "はいはい": "嗯嗯",
    "うん": "嗯",
    "うんうん": "嗯嗯",
    "ええ": "嗯",
    "えっ": "诶？",
    "え": "诶？",
    "え？": "诶？",
    "あ": "啊",
    "あっ": "啊",
    "あー": "啊——",
    "いや": "不",
    "いやいや": "不不",
    "いいえ": "不是",
    "そう": "是啊",
    "そうそう": "对对",
    "そうですね": "是啊",
    "そうだね": "是啊",
    "そうだよね": "是啊",
    "だよね": "对啊",
    "そうか": "这样啊",
    "たしかに": "确实",
    "なるほど": "原来如此",
    "へえ": "诶",
    "ほんと": "真的？",
    "ほんとう": "真的？",
    "本当": "真的？",
    "まじ": "真的？",
    "ごめん": "抱歉",
    "ごめんなさい": "对不起",
    "すみません": "不好意思",
    "ありがとう": "谢谢",
    "ありがとうございます": "谢谢",
    "おやすみ": "晚安",
    "おやすみなさい": "晚安",
    "いただきます": "我开动了",
    "ごちそうさま": "多谢款待",
    "ごちそうさまでした": "多谢款待",
    "ん": "嗯",
    "んー": "嗯……",
    "うーん": "嗯……",
    "ん？": "嗯？",
    "はぁ": "唉",
    "ふーん": "哦",
    "えーと": "那个……",
    "おい": "喂",
    "はーい": "好——",
    "よし": "好",
    "よしよし": "好了好了",
    "ちょっと": "等一下",
    "ちょっと待って": "等一下",
    "待って": "等一下",
}

# 译文侧的应答词：用于"不要把语气词并进上一句"的判断。
_CHINESE_INTERJECTIONS = frozenset(INTERJECTION_TRANSLATIONS.values()) | {
    "是",
    "是的",
    "不是",
    "好",
    "好的",
    "对",
    "对啊",
    "嗯",
    "嗯嗯",
    "哦",
    "噢",
    "哈",
    "哎",
    "唉",
    "啊",
    "诶",
    "谢谢",
    "抱歉",
    "对不起",
    "晚安",
    "没问题",
}

_INTERJECTION_SPLIT = re.compile(r"[、，,\s]+")
_INTERJECTION_TRIM = "。！？!?…「」『』（）()　 \t"


def _strip_for_match(text: str) -> str:
    return re.sub(r"[\s\u3000。、，．！？!?…「」『』（）()]", "", text)


def normalize_japanese(text: str) -> str:
    """轻度规范化：去零宽字符、压缩空白、删除首尾空白。

    不做 NFKC——那会把日语的 ``！？`` 折成半角，反而破坏排版。
    """
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = re.sub(r"[\u3000\t]+", " ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def normalize_chinese(text: str) -> str:
    """把机器翻译常见的半角标点/句首逗号/汉字间空格整理成中文排版。"""
    cleaned = normalize_japanese(text)

    # 汉字/假名之间的空格去掉
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff\u3040-\u30ff])\s+(?=[\u4e00-\u9fff\u3040-\u30ff])", "", cleaned)

    # 半角标点接到中日文后面时转全角
    cjk = r"[\u4e00-\u9fff\u3040-\u30ff]"
    cleaned = re.sub(rf"(?<={cjk})\s*,\s*", "，", cleaned)
    cleaned = re.sub(rf"(?<={cjk})\s*\.(?![0-9])", "。", cleaned)
    cleaned = re.sub(rf"(?<={cjk})\s*\?(?=[\s\u4e00-\u9fff]|$)", "？", cleaned)
    cleaned = re.sub(rf"(?<={cjk})\s*!(?=[\s\u4e00-\u9fff]|$)", "！", cleaned)
    cleaned = re.sub(rf"(?<={cjk}):", "：", cleaned)
    cleaned = re.sub(rf"(?<={cjk});", "；", cleaned)

    # 句首残留的标点（早期 seq2seq 翻译常输出 ",是的,是的."）
    cleaned = re.sub(r"^[\s，、。．,.;；:：!！?？]+", "", cleaned)
    # 句尾残留的逗号/顿号（按句切分后常见）
    cleaned = re.sub(r"[，、；,;]+$", "", cleaned)
    # 重复标点收敛
    cleaned = re.sub(r"([，。！？；：])\1+", r"\1", cleaned)
    return cleaned.strip()


def apply_renames(text: str, renames: Iterable[tuple[str, str]]) -> str:
    """按"旧中文写法 → 新中文写法"统一译名。

    这是给"不懂日语、只看中文"的用户准备的：翻译模型把某个名字译成
    "希卡尔"，用户完全可以只在中文层面把它改成"光"，不需要知道日语原词。
    长的旧写法先替换，避免短词抢在前面把长词拆坏。
    """
    updated = text
    for old, new in sorted(renames, key=lambda pair: len(pair[0]), reverse=True):
        if old and old != new:
            updated = updated.replace(old, new)
    return updated


def kana_ratio(text: str) -> float:
    """假名占比，用来发现"日语素材被识别成了中文/英文"的漂移。"""
    kana = len(re.findall(r"[\u3040-\u30ff]", text))
    meaningful = len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff0-9A-Za-z]", text))
    if meaningful == 0:
        return 0.0
    return kana / meaningful


def normalize_interjection_key(text: str) -> str:
    """去掉标点/括号/空白，得到用于查表的应答词键。"""
    stripped = normalize_japanese(text)
    return stripped.strip(_INTERJECTION_TRIM)


def translate_interjection(text: str) -> str | None:
    """整句就是应答词时返回固定中文译法，否则返回 ``None``。

    只做"整句命中"：句子里还有别的实词就交回给翻译模型，避免把
    「はい、わかりました」也硬翻成「嗯」。
    """
    raw = normalize_japanese(text).rstrip()
    key = normalize_interjection_key(text)
    if not key:
        return None
    # 先按原样查（「ん？」和「ん」语气不同），再按去标点后的键查。
    for candidate in (raw, key):
        direct = INTERJECTION_TRANSLATIONS.get(candidate)
        if direct is not None:
            return direct
    tokens = [token for token in _INTERJECTION_SPLIT.split(key) if token]
    if len(tokens) < 2:
        return None
    translated: list[str] = []
    for token in tokens:
        mapped = INTERJECTION_TRANSLATIONS.get(token)
        if mapped is None:
            return None
        translated.append(mapped)
    return "，".join(translated)


def is_interjection(text: str, *, max_chars: int = 6) -> bool:
    """判断一条字幕是不是纯粹的应答/语气词（中日两种写法都认）。"""
    key = normalize_interjection_key(text)
    if not key or len(key) > max_chars:
        return False
    if key in INTERJECTION_TRANSLATIONS or key in _CHINESE_INTERJECTIONS:
        return True
    tokens = [token for token in _INTERJECTION_SPLIT.split(key) if token]
    if len(tokens) < 2:
        return False
    return all(
        token in INTERJECTION_TRANSLATIONS or token in _CHINESE_INTERJECTIONS
        for token in tokens
    )


def is_hallucination(text: str) -> bool:
    """已知幻觉句式或整句只有标点。"""
    stripped = normalize_japanese(text)
    if _strip_for_match(stripped) == "":
        return True
    if stripped in _MEANINGLESS:
        return True
    compact = _strip_for_match(stripped)
    return any(pattern.match(compact) for pattern in _HALLUCINATION_PATTERNS)


def looks_like_language_drift(text: str) -> bool:
    """日语素材里整段没有假名（典型的中文漂移）时需要人工复核。"""
    stripped = normalize_japanese(text)
    if len(stripped) < 10:
        return False
    return kana_ratio(stripped) < 0.05


def drop_hallucinations(
    segments: Iterable[SubtitleSegment],
    *,
    language: str | None = None,
) -> list[SubtitleSegment]:
    kept: list[SubtitleSegment] = []
    for segment in segments:
        text = normalize_japanese(segment.text)
        if not text or is_hallucination(text):
            continue
        if language in {"ja", "jp"} and _is_all_latin(text) and len(text) < 40:
            # 日语素材里突然冒出一整句短英文，多半是幻觉/误识别
            continue
        kept.append(replace(segment, text=text))
    return kept


def dedupe_consecutive(
    segments: Iterable[SubtitleSegment],
    *,
    max_repeats: int = 2,
) -> list[SubtitleSegment]:
    """Whisper 的复读机：同一句连续出现 3 次以上只保留前两次。"""
    deduped: list[SubtitleSegment] = []
    run_key: str | None = None
    run_count = 0
    for segment in segments:
        key = _strip_for_match(segment.text)
        if key and key == run_key:
            run_count += 1
            if run_count > max_repeats:
                continue
        else:
            run_key = key
            run_count = 1
        deduped.append(segment)
    return deduped


def estimate_cue_duration(text: str, *, chars_per_second: float = 6.0, minimum: float = 1.0, maximum: float = 8.0) -> float:
    """按字数估算一条字幕的合理展示时长。"""
    chars = len(_strip_for_match(text)) or 1
    return max(minimum, min(maximum, chars / chars_per_second))


def repair_timestamps(
    segments: Iterable[SubtitleSegment],
    *,
    chars_per_second: float = 6.0,
    max_ratio: float = 2.5,
    max_duration: float = 10.0,
) -> list[SubtitleSegment]:
    """修正明显异常的时间戳。

    Whisper 在音乐/环境音上偶尔会把一两句台词标成几分钟甚至几十分钟长。
    这类时间戳不仅让字幕长时间挂在屏幕上，还会污染后续的"按句合并"：
    异常片段的 end 远在将来，后面的句子都被吸进同一个翻译单元。
    """
    items = list(segments)
    repaired: list[SubtitleSegment] = []
    for index, segment in enumerate(items):
        start = max(0.0, segment.start)
        expected = estimate_cue_duration(segment.text, chars_per_second=chars_per_second)
        duration = max(0.2, segment.end - start)
        end = start + expected if duration > max(max_duration, expected * max_ratio) else segment.end

        next_start = items[index + 1].start if index + 1 < len(items) else None
        if next_start is not None and start < next_start < end:
            end = next_start
        repaired.append(
            replace(segment, start=round(start, 3), end=round(max(start + 0.2, end), 3))
        )
    return repaired


def merge_short_segments(
    segments: Iterable[SubtitleSegment],
    *,
    min_duration: float = 1.0,
    max_gap: float = 0.8,
    max_chars: int = 40,
    protect_replies: bool = True,
) -> list[SubtitleSegment]:
    """把过短且紧贴上一句的碎片并入上一句（只合并，不跨大停顿）。

    ``protect_replies`` 会保护应答词：「はい」被翻译成「嗯」之后只有 1 个字，
    一旦并进上一句，就变成"别的话+嗯"，对话的节奏和说话人都被抹掉了。
    """
    merged: list[SubtitleSegment] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap = segment.start - previous.end
        combined_len = len(previous.text) + len(segment.text)
        if protect_replies and (is_interjection(previous.text) or is_interjection(segment.text)):
            merged.append(segment)
            continue
        if (
            gap <= max_gap
            and combined_len <= max_chars
            and (previous.end - previous.start) < min_duration
        ):
            merged[-1] = replace(
                previous,
                end=max(previous.end, segment.end),
                text=f"{previous.text}{segment.text}",
                source_text=_join_source(previous, segment),
            )
        else:
            merged.append(segment)
    return merged


def split_long_segments(
    segments: Iterable[SubtitleSegment],
    *,
    max_chars: int = 20,
    max_duration: float = 8.0,
) -> list[SubtitleSegment]:
    """把超长字幕按标点拆成多条，并按字符数分配时间。"""
    result: list[SubtitleSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if len(text) <= max_chars and (segment.end - segment.start) <= max_duration:
            result.append(segment)
            continue

        pieces = _split_text(text, max_chars=max_chars)
        if len(pieces) <= 1:
            result.append(segment)
            continue

        total_chars = sum(len(piece) for piece in pieces) or 1
        span = max(0.0, segment.end - segment.start)
        cursor = segment.start
        for index, piece in enumerate(pieces):
            share = max(0.4, span * len(piece) / total_chars)
            end = segment.end if index == len(pieces) - 1 else min(segment.end, cursor + share)
            result.append(
                replace(
                    segment,
                    start=round(cursor, 3),
                    end=round(max(cursor + 0.3, end), 3),
                    text=piece,
                )
            )
            cursor = end
    return result


def enforce_timing(
    segments: Iterable[SubtitleSegment],
    *,
    min_duration: float = 0.8,
    short_text_chars: int = 3,
    short_min_duration: float = 1.2,
) -> list[SubtitleSegment]:
    """保证每条字幕有最短展示时长，且不跟下一条重叠。

    极短的字幕（「嗯」「啊」）只有 1–2 个字，按 0.8 秒闪过去几乎看不见，
    因此单独给一个更长的下限。
    """
    items = list(segments)
    fixed: list[SubtitleSegment] = []
    for index, segment in enumerate(items):
        start = max(0.0, segment.start)
        end = max(segment.end, start + 0.2)
        floor = min_duration
        if len(_strip_for_match(segment.text)) <= short_text_chars:
            floor = max(floor, short_min_duration)
        if end - start < floor:
            end = start + floor
        next_start = items[index + 1].start if index + 1 < len(items) else None
        if next_start is not None and end > next_start:
            end = max(start + 0.2, next_start - 0.01)
        fixed.append(replace(segment, start=round(start, 3), end=round(end, 3)))
    return fixed


def finalize_subtitles(
    segments: Iterable[SubtitleSegment],
    *,
    language: str | None = None,
    max_chars: int = 20,
    max_duration: float = 8.0,
    min_duration: float = 0.8,
    chars_per_second: float = 6.0,
) -> list[SubtitleSegment]:
    """对最终写入字幕的文本执行完整清洗流程。"""
    pipeline = list(segments)
    pipeline = repair_timestamps(pipeline, chars_per_second=chars_per_second)
    pipeline = drop_hallucinations(pipeline, language=language)
    pipeline = dedupe_consecutive(pipeline)
    pipeline = merge_short_segments(pipeline)
    pipeline = split_long_segments(pipeline, max_chars=max_chars, max_duration=max_duration)
    pipeline = enforce_timing(pipeline, min_duration=min_duration)
    return pipeline


def build_quality_report(
    segments: Iterable[SubtitleSegment],
    *,
    language: str | None = None,
) -> list[dict]:
    """诊断用的逐条对照表（原文 / 译文 / 可疑标记）。

    输入应当是翻译后的片段：``text`` 是中文译文，``source_text`` 是原文。
    句级合并会让片段数量变少，所以不能再按原始识别片段的下标配对。
    """
    rows: list[dict] = []
    for index, segment in enumerate(segments):
        source = normalize_japanese(segment.source_text or "")
        flags: list[str] = []
        if source and is_hallucination(source):
            flags.append("hallucination")
        if language in {"ja", "jp"} and source and looks_like_language_drift(source):
            flags.append("low_kana")
        if len(source) > 40:
            flags.append("very_long")
        if segment.end - segment.start < 0.5:
            flags.append("very_short")
        rows.append(
            {
                "index": index + 1,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "source": source,
                "text": segment.text,
                "kana_ratio": round(kana_ratio(source), 3) if source else 0.0,
                "flags": flags,
            }
        )
    return rows


def _join_source(previous: SubtitleSegment, segment: SubtitleSegment) -> str | None:
    if previous.source_text and segment.source_text:
        return f"{previous.source_text}{segment.source_text}"
    return previous.source_text or segment.source_text


def _is_all_latin(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text)
    others = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", text)
    return bool(letters) and not others


def _split_text(text: str, *, max_chars: int) -> list[str]:
    """优先在标点处断句，凑不够时按字数硬切。"""
    pieces: list[str] = []
    remainder = text
    while len(remainder) > max_chars:
        window = remainder[: max_chars + 1]
        cut = max(
            (window.rfind(mark) + 1 for mark in "。！？!?、，,；;…"),
            default=-1,
        )
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(remainder[:cut].strip())
        remainder = remainder[cut:].lstrip()
    if remainder.strip():
        pieces.append(remainder.strip())
    return [piece for piece in pieces if piece]
