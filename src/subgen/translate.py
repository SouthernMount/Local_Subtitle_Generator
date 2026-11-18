"""翻译层：本地 GGUF 大模型（llama.cpp）把源语言字幕翻成简体中文。

NLLB 这类 600M 级 seq2seq 模型逐句直译、不看上下文，实测会把「先輩」翻成
“先生”、把「クリーニング」翻成“干净”，在字幕场景里不可用。现在改用本地
指令微调模型，逐句翻译时带上前面两轮原文与译文作为上下文，并把术语表
直接写进提示词。

模型与依赖全部本地：GGUF 放在 ``models/`` 下，推理走 llama.cpp
（``llama-cpp-python``），不访问网络。
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from .glossary import GlossaryEntry, describe_glossary
from .postprocess import (
    is_interjection,
    normalize_chinese,
    normalize_japanese,
    translate_interjection,
)
from .transcribe import TranscriptSegment, cuda_device_count

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TRANSLATION_MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-7b-instruct-q4_k_m"

# 单次送进翻译模型的原文长度上限（字符）。整句一起翻远比逐条翻译可读，
# 但对白场景里过长的合并会把多个说话人揉进同一条字幕，反而破坏连贯性。
MAX_GROUP_CHARS = 60
# 上下文轮数：带上前两句原文和译文，足以稳住称呼和代词，又不会挤占上下文窗口。
DEFAULT_CONTEXT_PAIRS = 2
DEFAULT_N_CTX = 2048
# 一次请求翻译多少句。批量请求能把相邻台词一起送进去，跨句的指代更稳，
# 请求数也少一个数量级；但行数越多，模型漏行/错位的概率越高，因此有失败就回落逐句。
DEFAULT_BATCH_SIZE = 1
# 单句译文长度上限倍数：超过这个比例的产出通常是模型在"解释"而不是翻译。
_MAX_LENGTH_RATIO = 4

# 句末标记：命中这些字符就认为一句话结束。
_SENTENCE_ENDINGS = "。！？!?…」』）)"
# 典型的话轮起始词：出现时即使上一句没以句号结尾也另起一条，
# 避免把"应答 + 下一句"翻成一句莫名其妙的话。
_TURN_STARTERS = (
    "はい", "いいえ", "いや", "えっ", "え、", "あ、", "あの", "そう", "うん", "んー",
    "じゃ", "でも", "だから", "ちょっと", "ありがとう", "すいません", "すみません",
    "ごめん", "おはよう", "こんにちは", "こんばんは", "失礼",
)

CHINESE_LANGUAGE_CODES = {
    "yue",
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zh-tw",
    "zho_hans",
    "zho_hant",
}

LANGUAGE_NAMES = {
    "ja": "日语",
    "jp": "日语",
    "en": "英语",
    "ko": "韩语",
    "th": "泰语",
    "vi": "越南语",
    "ar": "阿拉伯语",
    "de": "德语",
    "es": "西班牙语",
    "fr": "法语",
    "hi": "印地语",
    "it": "意大利语",
    "pt": "葡萄牙语",
    "ru": "俄语",
    "yue": "粤语",
}

_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
# 模型偶尔会把"这句话的意思是…"这类说明也写进译文，逐字删掉这些尾巴。
_EXPLANATION_MARKERS = ("这句话", "意思是", "根据上下文", "原文是", "日语原文", "（注", "(注")
_PREFIX = re.compile(r"^(译文|中文|简体中文|翻译)[:：]\s*")
# 模型有时把对话破折号也抄进来，单条字幕里这是纯噪声。
_LEADING_DASH = re.compile(r"^[-—–]+\s*")
_TRIM_CHARS = " \t\"'“”「」『』"


class TranslationError(RuntimeError):
    """Raised when local Chinese translation cannot be completed."""


ProgressCallback = Callable[[dict], None]


@dataclass(frozen=True)
class _TranslationStats:
    total: int = 0
    failed: int = 0


def resolve_default_translation_model() -> str:
    """翻译模型路径：环境变量 > 项目 models 目录。"""
    env_path = os.environ.get("SUBGEN_TRANSLATION_MODEL")
    if env_path:
        return env_path
    return str(LOCAL_TRANSLATION_MODEL_DIR)


DEFAULT_TRANSLATION_MODEL = resolve_default_translation_model()


def resolve_model_file(model_path: str | Path) -> Path:
    """把"目录或文件"解析成实际要加载的 GGUF 分片。

    目录里允许放单个 ``*.gguf``，也允许放 llama.cpp 的命名分片
    （``...-00001-of-00002.gguf``）——后者只需要指向第一个分片，其余会自动加载。
    """
    path = Path(model_path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        first_shard = sorted(path.glob("*-00001-of-*.gguf"))
        if first_shard:
            return first_shard[0]
        candidates = sorted(path.glob("*.gguf"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = "、".join(item.name for item in candidates)
            raise TranslationError(
                f"目录里有多个 GGUF 文件，无法确定用哪个：{path}（{names}）。"
                "请用 --translation-model 直接指定要加载的文件。"
            )
    raise TranslationError(
        f"找不到本地翻译模型：{path}。"
        "请把 GGUF 模型放到 models/qwen2.5-7b-instruct-q4_k_m/，"
        "或用 --translation-model 指定文件/目录。"
    )


def verify_translation_model(
    model_path: str | Path = DEFAULT_TRANSLATION_MODEL,
) -> Path:
    """确认本地 GGUF 存在且可定位；翻译模型不需要下载。"""
    return resolve_model_file(model_path)


def translate_segments_to_chinese(
    segments: Iterable[TranscriptSegment],
    *,
    source_language: str | None,
    model_path: str | Path = DEFAULT_TRANSLATION_MODEL,
    device: str = "auto",
    progress_callback: ProgressCallback | None = None,
    glossary: Sequence[GlossaryEntry] | None = None,
    context_pairs: int = DEFAULT_CONTEXT_PAIRS,
    n_ctx: int = DEFAULT_N_CTX,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[TranscriptSegment]:
    """把识别出的片段翻成简体中文。

    与逐条翻译不同，这里先把连续片段按句末标点合并成完整句子再翻译，
    再把译文分配回对应的字幕时间轴，从而兼顾上下文与时间戳。

    应答词（「はい」「うん」）直接查表，不送进模型——模型会把它们扩写成
    「好的，我明白了」这类完整句，字幕长度和语气都不对。
    """
    prepared_segments = list(segments)
    if not prepared_segments:
        _emit_progress(
            progress_callback,
            stage="translate",
            message="没有可翻译的字幕片段",
            percent=0.8,
            current=0,
            total=0,
        )
        return []

    if _is_chinese_language(source_language):
        _emit_progress(
            progress_callback,
            stage="translate",
            message="源语言已是中文，跳过翻译",
            percent=0.85,
            current=len(prepared_segments),
            total=len(prepared_segments),
        )
        return prepared_segments

    groups = _group_segments(prepared_segments, max_chars=MAX_GROUP_CHARS)
    entries = list(glossary or ())

    texts_by_index: dict[int, str] = {}
    pending: list[tuple[int, str]] = []
    for index, group in enumerate(groups):
        unit_text = "".join(normalize_japanese(segment.text) for segment in group)
        if len(group) == 1:
            mapped = translate_interjection(unit_text)
            if mapped is not None:
                texts_by_index[index] = mapped
                continue
        pending.append((index, unit_text))

    if pending:
        stats = _translate_pending(
            pending,
            groups=groups,
            source_language=source_language,
            model_path=model_path,
            device=device,
            progress_callback=progress_callback,
            glossary=entries,
            context_pairs=context_pairs,
            n_ctx=n_ctx,
            batch_size=batch_size,
            total_segments=len(prepared_segments),
            output=texts_by_index,
        )
        if stats.failed and stats.failed >= stats.total:
            raise TranslationError(
                f"全部 {stats.total} 条台词都没能翻译成中文，请检查翻译模型与提示词设置。"
            )

    translated_segments: list[TranscriptSegment] = []
    for index, group in enumerate(groups):
        joined_source = "".join(
            (segment.source_text or segment.text) for segment in group
        )
        text = normalize_chinese(texts_by_index.get(index, ""))
        if not text:
            continue
        translated_segments.append(
            TranscriptSegment(
                start=group[0].start,
                end=group[-1].end,
                text=text,
                source_text=joined_source if _group_has_source(group) else None,
            )
        )
    return translated_segments


def _translate_pending(
    pending: Sequence[tuple[int, str]],
    *,
    groups: Sequence[Sequence[TranscriptSegment]],
    source_language: str | None,
    model_path: str | Path,
    device: str,
    progress_callback: ProgressCallback | None,
    glossary: Sequence[GlossaryEntry],
    context_pairs: int,
    n_ctx: int,
    batch_size: int,
    total_segments: int,
    output: dict[int, str],
) -> _TranslationStats:
    model_file = resolve_model_file(model_path)
    _emit_progress(
        progress_callback,
        stage="load_translation_model",
        message=f"正在加载本地翻译模型 {model_file.name}",
        percent=0.6,
    )
    llm = _load_llm(model_file, device=device, n_ctx=n_ctx)
    system_prompt = _build_system_prompt(source_language, glossary)

    total = len(pending)
    started_at = monotonic()
    processed_segments = 0
    failed = 0
    history: list[tuple[str, str]] = []
    step = max(1, batch_size)
    _emit_progress(
        progress_callback,
        stage="translate",
        message=f"正在翻译字幕 0 / {total_segments}",
        percent=0.65,
        current=0,
        total=total_segments,
        eta_seconds=None,
    )

    for offset in range(0, total, step):
        chunk = pending[offset : offset + step]
        results = _translate_batch(
            llm,
            system_prompt=system_prompt,
            source_texts=[text for _, text in chunk],
            history=history if context_pairs > 0 else [],
            glossary=glossary,
        )
        for (index, source_text), translated in zip(chunk, results):
            if not _is_usable(translated, source_text):
                # 批量请求漏行、错位或格式不合格时，这一句退回逐句重译。
                translated = _translate_one(
                    llm,
                    system_prompt=system_prompt,
                    source_text=source_text,
                    history=history if context_pairs > 0 else [],
                    glossary=glossary,
                )
            if translated:
                history.append((source_text, translated))
                if len(history) > context_pairs:
                    history.pop(0)
                output[index] = translated
            else:
                failed += 1

            processed_segments += len(groups[index])
            elapsed = monotonic() - started_at
            _emit_progress(
                progress_callback,
                stage="translate",
                message=f"正在翻译字幕 {min(processed_segments, total_segments)} / {total_segments}",
                percent=0.65 + (0.25 * min(processed_segments, total_segments) / max(total_segments, 1)),
                current=min(processed_segments, total_segments),
                total=total_segments,
                elapsed_seconds=elapsed,
                eta_seconds=_estimate_remaining_seconds(
                    elapsed_seconds=elapsed,
                    current=processed_segments,
                    total=total_segments,
                ),
            )

    if failed:
        _emit_progress(
            progress_callback,
            stage="warning",
            message=f"翻译完成，有 {failed} 条台词未能可靠翻译，已跳过",
            percent=0.9,
        )
    return _TranslationStats(total=total, failed=failed)


def _load_llm(model_file: Path, *, device: str, n_ctx: int):
    try:
        from llama_cpp import Llama
    except ModuleNotFoundError as exc:
        raise TranslationError(
            "缺少依赖 llama-cpp-python。Windows 上请先安装项目自带的预编译 wheel："
            "models\\runtime\\llama_cpp_python-*.whl，再运行 "
            '`python -m pip install -e ".[translate]"`。'
        ) from exc

    n_gpu_layers = -1 if _use_gpu(device) else 0
    try:
        return Llama(
            model_path=str(model_file),
            chat_format="chatml",
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - 真实模型加载失败
        raise TranslationError(
            f"本地翻译模型加载失败：{model_file}。请确认文件完整、显存足够。"
            f"原因：{exc}"
        ) from exc


def _use_gpu(device: str) -> bool:
    normalized = (device or "auto").strip().lower()
    if normalized in {"cuda", "gpu"}:
        return True
    if normalized == "cpu":
        return False
    return cuda_device_count() > 0


def _build_system_prompt(
    source_language: str | None,
    glossary: Sequence[GlossaryEntry],
) -> str:
    language_name = LANGUAGE_NAMES.get(
        (source_language or "").strip().lower().split("-")[0],
        "原文",
    )
    rules = [
        f"你是专业的{language_name}影视字幕翻译，把台词翻译成简体中文。",
        "规则：",
        "1. 只输出译文本身：不要解释、不要加引号、不要保留原文、不要输出多行。",
        "2. 口语化、忠实原意，不要扩写，不要合并成多句。",
        "3. 保持人名、称呼、代词与上文一致。",
    ]
    if glossary:
        lines = [f"{entry.source}→{entry.target}" for entry in glossary[:40]]
        rules.append("4. 术语表（必须按给定写法翻译）：\n" + "\n".join(lines))
    return "\n".join(rules)


def _build_user_prompt(
    source_text: str,
    history: Sequence[tuple[str, str]],
) -> str:
    if not history:
        return f"台词：\n{source_text}\n\n只输出这一句的中文译文。"
    context = "\n".join(
        f"原文：{source[:60]}\n译文：{translated[:60]}" for source, translated in history
    )
    return (
        f"上文（只帮助理解，不要翻译）：\n{context}\n\n"
        f"台词：\n{source_text}\n\n只输出这一句的中文译文。"
    )


def _build_batch_prompt(texts: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(texts, start=1))
    return (
        f"台词（共 {len(texts)} 句，按顺序编号）：\n{numbered}\n\n"
        f"按同样的编号输出 {len(texts)} 行简体中文译文，一行一句，不要解释。"
    )


def _build_messages(
    system_prompt: str,
    texts: Sequence[str],
    history: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    if len(texts) == 1:
        user = _build_user_prompt(texts[0], history)
    elif history:
        context = "\n".join(
            f"原文：{source[:60]}\n译文：{translated[:60]}" for source, translated in history
        )
        user = f"上文（只帮助理解，不要翻译）：\n{context}\n\n{_build_batch_prompt(texts)}"
    else:
        user = _build_batch_prompt(texts)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def _chat(llm, messages: list[dict[str, str]], *, max_tokens: int, temperature: float) -> str:
    try:
        completion = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            repeat_penalty=1.05,
        )
    except Exception as exc:  # pragma: no cover - llama.cpp 运行时异常
        raise TranslationError(f"本地翻译模型推理失败：{exc}") from exc
    return completion["choices"][0]["message"]["content"] or ""


# 带人名的句子常被原样留下假名（ヒカルちゃん…），第一次校验会判不合格。
# 重试时明确要求人名音译：既不丢整句，也不把日文名混进中文。
_RETRY_INSTRUCTION = (
    "只输出简体中文译文，不要任何解释或引号；"
    "不要保留任何日文假名，人名音译成中文（例如 ヒカル→希卡尔、みえ→美惠）。"
)


def _missing_glossary_terms(
    text: str,
    source_text: str,
    glossary: Sequence[GlossaryEntry],
) -> list[GlossaryEntry]:
    """原文命中、译文却没用到指定写法的术语。

    术语表只是提示词的一部分时，模型经常"看到但不遵守"；这个检查把
    "该用没用"变成可重试的失败，而不是静默放过。
    """
    if not text or not glossary:
        return []
    return [
        entry
        for entry in glossary
        if entry.source and entry.source in source_text and entry.target not in text
    ]


def _translate_one(
    llm,
    *,
    system_prompt: str,
    source_text: str,
    history: Sequence[tuple[str, str]],
    glossary: Sequence[GlossaryEntry] = (),
) -> str:
    """翻译一句；格式不合格或术语未命中时换更明确的提示重试一次。"""
    messages = _build_messages(system_prompt, [source_text], history)
    best = ""
    for attempt in range(2):
        raw = _chat(
            llm,
            messages,
            max_tokens=96,
            temperature=0.2 if attempt == 0 else 0.1,
        )
        cleaned = _clean_translation(raw)
        missing = _missing_glossary_terms(cleaned, source_text, glossary)
        if _is_usable(cleaned, source_text):
            best = cleaned
            if not missing:
                return cleaned
        messages.append({"role": "assistant", "content": raw})
        instruction = _RETRY_INSTRUCTION
        if missing:
            pairs = "、".join(f"{entry.source}→{entry.target}" for entry in missing[:5])
            instruction += f"术语表要求：{pairs}，译文必须使用这些写法。"
        messages.append({"role": "user", "content": instruction})
    # 重试后仍可能没命中术语：正文可用就保留，宁可译名不统一也不整条丢掉。
    return best


_NUMBERED_LINE = re.compile(r"^\s*[（(]?(\d{1,2})[)）.、:：]\s*(.+?)\s*$")


def _parse_batch_output(raw: str, count: int) -> list[str] | None:
    lines = [line for line in raw.splitlines() if line.strip()]
    numbered: dict[int, str] = {}
    for line in lines:
        match = _NUMBERED_LINE.match(line)
        if match:
            numbered[int(match.group(1))] = match.group(2)
    if numbered and len(numbered) >= count:
        return [numbered.get(index, "") for index in range(1, count + 1)]
    if len(lines) == count:
        return lines
    return None


def _translate_batch(
    llm,
    *,
    system_prompt: str,
    source_texts: Sequence[str],
    history: Sequence[tuple[str, str]],
    glossary: Sequence[GlossaryEntry],
) -> list[str]:
    """一次请求翻多句；行数对不上就返回空串，由调用方逐句重译。"""
    if len(source_texts) == 1:
        return [
            _translate_one(
                llm,
                system_prompt=system_prompt,
                source_text=source_texts[0],
                history=history,
                glossary=glossary,
            )
        ]
    messages = _build_messages(system_prompt, source_texts, history)
    raw = _chat(
        llm,
        messages,
        max_tokens=min(640, 96 * len(source_texts)),
        temperature=0.1,
    )
    parsed = _parse_batch_output(raw, len(source_texts))
    if parsed is None:
        return [""] * len(source_texts)
    return [_clean_translation(line) for line in parsed]


def _clean_translation(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    # 只保留第一行：模型有时会先给译文、再补一段说明。
    text = text.splitlines()[0]
    text = _PREFIX.sub("", text)
    text = _LEADING_DASH.sub("", text)
    for marker in _EXPLANATION_MARKERS:
        position = text.find(marker)
        if position > 0:
            text = text[:position]
    return text.strip(_TRIM_CHARS).strip()


def _is_usable(text: str, source_text: str) -> bool:
    if not text:
        return False
    if _KANA.search(text):
        return False
    # 偶尔会漏出英文单词（「妈妈serious吗？」），中文比英文还少就不能当译文用；
    # 但 "OK"、"SNS" 这类夹在中文里的少量拉丁字母是正常的。
    latin = len(_LATIN.findall(text))
    if latin and len(_CJK.findall(text)) < latin:
        return False
    limit = max(12, len(source_text) * _MAX_LENGTH_RATIO)
    return len(text) <= limit


def _group_segments(
    segments: Sequence[TranscriptSegment],
    *,
    max_chars: int = MAX_GROUP_CHARS,
    max_gap: float = 1.2,
    min_interjection_gap: float = 0.25,
) -> list[list[TranscriptSegment]]:
    """把连续片段聚成"一句话"的翻译单元。

    合并的条件很克制：间隔要小、总长度有限，而且下一句不能是典型的话轮起始
    词——否则多人对话会被揉成一条，译文就会答非所问。

    应答词（「はい」「うん」）单独成组：并进上一句后既丢了"这就是一句回答"的
    信息，也拿不到固定的中文译法。但只在前/后有停顿或带句末标点时才拆——连续
    说出的「そう思う」是一个句子，不能把「そう」单独拆出来。
    """
    segments = list(segments)
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        if current:
            gap = segment.start - current[-1].end
            length = sum(len(item.text) for item in current) + len(segment.text)
            if (
                gap > max_gap
                or length > max_chars
                or _starts_new_turn(segment.text)
            ):
                groups.append(current)
                current = []
        if _is_standalone_interjection(segments, index, min_gap=min_interjection_gap):
            if current:
                groups.append(current)
                current = []
            groups.append([segment])
            continue
        current.append(segment)
        if segment.text.rstrip().endswith(tuple(_SENTENCE_ENDINGS)):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _is_standalone_interjection(
    segments: Sequence[TranscriptSegment],
    index: int,
    *,
    min_gap: float = 0.25,
) -> bool:
    """判断某个片段是不是"单独的一句应答词"。

    判据优先级：整句带句末标点 → 是；否则看前/后的静音间隔，有停顿才当成
    独立应答（连续说出的「そう + 思う」不算）。
    """
    segment = segments[index]
    if not is_interjection(segment.text):
        return False
    text = normalize_japanese(segment.text).rstrip()
    if text.endswith(("。", "！", "？", "!", "?", "…")):
        return True
    gaps: list[float] = []
    if index > 0:
        gaps.append(segment.start - segments[index - 1].end)
    if index + 1 < len(segments):
        gaps.append(segments[index + 1].start - segment.end)
    if not gaps:
        return True
    return max(gaps) >= min_gap


def _starts_new_turn(text: str) -> bool:
    stripped = text.lstrip(" 「『『（(")
    return any(stripped.startswith(word) for word in _TURN_STARTERS)


def _group_has_source(group: Sequence[TranscriptSegment]) -> bool:
    return any(segment.source_text for segment in group)


def _is_chinese_language(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().replace("_", "-").lower()
    return normalized in CHINESE_LANGUAGE_CODES or normalized.replace("-", "_") in CHINESE_LANGUAGE_CODES


def _estimate_remaining_seconds(
    *,
    elapsed_seconds: float,
    current: int,
    total: int,
) -> float | None:
    if current <= 0 or total <= 0 or current >= total:
        return None
    seconds_per_item = elapsed_seconds / current
    return max(0.0, seconds_per_item * (total - current))


def _emit_progress(
    progress_callback: ProgressCallback | None,
    **payload,
) -> None:
    if progress_callback is not None:
        progress_callback(payload)
