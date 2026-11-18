import re
import sys
import types
from pathlib import Path

import pytest

from subgen.glossary import GlossaryEntry
from subgen.transcribe import TranscriptSegment
from subgen.translate import (
    TranslationError,
    _estimate_remaining_seconds,
    _group_segments,
    _build_batch_prompt,
    _build_messages,
    _missing_glossary_terms,
    _parse_batch_output,
    _translate_batch,
    _translate_one,
    resolve_model_file,
    translate_segments_to_chinese,
)


def _extract_source(messages: list[dict]) -> str:
    """从提示词里取出待翻译的原文（重试时 messages[1] 仍是原始请求）。"""
    match = re.search(r"台词：\n(.*?)\n\n只输出", messages[1]["content"], re.S)
    return match.group(1) if match else ""


def install_fake_llm(monkeypatch: pytest.MonkeyPatch, script: dict[str, list[str]]):
    class FakeLlama:
        calls: list[tuple[list[dict], str]] = []
        init_kwargs: dict = {}

        def __init__(self, **kwargs):
            FakeLlama.init_kwargs = kwargs

        def create_chat_completion(self, messages, **_kwargs):
            source = _extract_source(messages)
            FakeLlama.calls.append((messages, source))
            queue = script.get(source) or []
            text = queue.pop(0) if queue else source
            return {"choices": [{"message": {"content": text}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
    return FakeLlama


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"gguf")
    return path


def test_translate_skips_chinese_source_language(tmp_path: Path) -> None:
    segments = [TranscriptSegment(0, 1, "已经是中文。")]

    assert translate_segments_to_chinese(segments, source_language="zh") == segments


def test_translate_empty_segments_returns_empty() -> None:
    assert translate_segments_to_chinese([], source_language="ja") == []


def test_translate_interjection_skips_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    def fail_loading(*_args, **_kwargs):
        raise AssertionError("应答词不应该加载翻译模型")

    monkeypatch.setattr("subgen.translate._load_llm", fail_loading)

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "はい。"), TranscriptSegment(1, 2, "うん")],
        source_language="ja",
        model_path=model_file,
    )

    assert translated == [
        TranscriptSegment(0, 1, "嗯"),
        TranscriptSegment(1, 2, "嗯"),
    ]


def test_translate_uses_local_llm_and_passes_context(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    script = {
        "今日は寒いですね。": ["今天真冷啊。"],
        "明日は晴れるそうです。": ["听说明天会放晴。"],
    }
    fake = install_fake_llm(monkeypatch, script)

    translated = translate_segments_to_chinese(
        [
            TranscriptSegment(0, 1, "今日は寒いですね。"),
            TranscriptSegment(2, 3, "明日は晴れるそうです。"),
        ],
        source_language="ja",
        model_path=model_file,
        device="cpu",
    )

    assert [segment.text for segment in translated] == ["今天真冷啊。", "听说明天会放晴。"]
    assert fake.init_kwargs["n_gpu_layers"] == 0
    assert fake.init_kwargs["chat_format"] == "chatml"
    # 第二句带上了第一句的原文和译文作为上下文
    assert "原文：今日は寒いですね。" in fake.calls[1][0][1]["content"]


def test_translate_strips_explanation_from_output(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    install_fake_llm(
        monkeypatch,
        {"こんにちは。": ["你好。这句话的意思是打招呼。"]},
    )

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "こんにちは。")],
        source_language="ja",
        model_path=model_file,
    )

    assert [segment.text for segment in translated] == ["你好。"]


def test_translate_strips_dialogue_dash(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    install_fake_llm(monkeypatch, {"了解です。": ["- 明白了。"]})

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "了解です。")],
        source_language="ja",
        model_path=model_file,
    )

    assert [segment.text for segment in translated] == ["明白了。"]


def test_translate_retries_when_output_keeps_source_language(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    fake = install_fake_llm(
        monkeypatch,
        {"おはよう。": ["おはよう。", "早上好。"]},
    )

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "おはよう。")],
        source_language="ja",
        model_path=model_file,
    )

    assert [segment.text for segment in translated] == ["早上好。"]
    assert len(fake.calls) == 2


def test_translate_retries_when_output_is_mostly_latin(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    fake = install_fake_llm(
        monkeypatch,
        {"マジ?": ["妈妈serious吗？", "真的吗？"]},
    )

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "マジ?")],
        source_language="ja",
        model_path=model_file,
    )

    assert [segment.text for segment in translated] == ["真的吗？"]
    assert len(fake.calls) == 2


def test_translate_all_failures_raise(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    install_fake_llm(
        monkeypatch,
        {"テスト。": ["テスト。", "テスト。"]},
    )

    with pytest.raises(TranslationError, match="都没能翻译"):
        translate_segments_to_chinese(
            [TranscriptSegment(0, 1, "テスト。")],
            source_language="ja",
            model_path=model_file,
        )


def test_translate_drops_single_failed_line(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    install_fake_llm(
        monkeypatch,
        {
            "テスト。": ["テスト。", "テスト。"],
            "本番です。": ["正式开始了。"],
        },
    )

    translated = translate_segments_to_chinese(
        [
            TranscriptSegment(0, 1, "テスト。"),
            TranscriptSegment(2, 3, "本番です。"),
        ],
        source_language="ja",
        model_path=model_file,
    )

    assert [segment.text for segment in translated] == ["正式开始了。"]


def test_translate_puts_glossary_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    model_file: Path,
) -> None:
    fake = install_fake_llm(monkeypatch, {"ヒカルだ。": ["是光。"], "ヒカル": ["光"]})

    translate_segments_to_chinese(
        [TranscriptSegment(0, 1, "ヒカルだ。")],
        source_language="ja",
        model_path=model_file,
        glossary=[GlossaryEntry("ヒカル", "光")],
    )

    system_prompt = fake.calls[0][0][0]["content"]
    assert "ヒカル→光" in system_prompt


def test_translate_marks_source_text(monkeypatch: pytest.MonkeyPatch, model_file: Path) -> None:
    install_fake_llm(monkeypatch, {"今日は寒いですね。": ["今天真冷啊。"]})

    translated = translate_segments_to_chinese(
        [TranscriptSegment(0, 2, "今日は寒いですね。", source_text="今日は寒いですね。")],
        source_language="ja",
        model_path=model_file,
    )

    assert translated[0].source_text == "今日は寒いですね。"


def test_resolve_model_file_accepts_single_gguf(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"gguf")

    assert resolve_model_file(tmp_path) == path
    assert resolve_model_file(path) == path


def test_resolve_model_file_prefers_first_shard(tmp_path: Path) -> None:
    first = tmp_path / "qwen-00001-of-00002.gguf"
    second = tmp_path / "qwen-00002-of-00002.gguf"
    first.write_bytes(b"1")
    second.write_bytes(b"2")

    assert resolve_model_file(tmp_path) == first


def test_resolve_model_file_rejects_ambiguous_directory(tmp_path: Path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"1")
    (tmp_path / "b.gguf").write_bytes(b"2")

    with pytest.raises(TranslationError, match="多个 GGUF"):
        resolve_model_file(tmp_path)


def test_resolve_model_file_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(TranslationError, match="找不到本地翻译模型"):
        resolve_model_file(tmp_path / "missing")


def test_estimate_remaining_seconds() -> None:
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=5, total=10) == 20
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=10, total=10) is None
    assert _estimate_remaining_seconds(elapsed_seconds=20, current=0, total=10) is None


def test_group_segments_merges_mid_sentence_fragments() -> None:
    segments = [
        TranscriptSegment(0, 1, "これは、その、"),
        TranscriptSegment(1, 2, "難しいですね"),
    ]

    assert len(_group_segments(segments)) == 1


def test_group_segments_splits_on_turn_starter() -> None:
    segments = [
        TranscriptSegment(0, 1, "今日は寒いですね"),
        TranscriptSegment(1, 2, "はい、そうですね。"),
    ]

    assert len(_group_segments(segments)) == 2


def test_group_segments_splits_on_long_gap() -> None:
    segments = [
        TranscriptSegment(0, 1, "今日は寒いですね"),
        TranscriptSegment(5, 6, "明日は晴れるそうです"),
    ]

    assert len(_group_segments(segments)) == 2


def test_group_segments_keeps_paused_reply_separate() -> None:
    segments = [
        TranscriptSegment(0, 1.5, "今日は寒いですね"),
        TranscriptSegment(1.9, 2.4, "はい。"),
        TranscriptSegment(2.5, 4.0, "明日は晴れるそうです。"),
    ]

    groups = _group_segments(segments)

    assert [len(group) for group in groups] == [1, 1, 1]


def test_group_segments_does_not_split_continuous_fragment() -> None:
    # 「そう」+「思う」连续说出时是一个句子，不能把「そう」单独当成应答词。
    segments = [
        TranscriptSegment(0, 1, "そう"),
        TranscriptSegment(1, 2, "思うんですけど。"),
    ]

    groups = _group_segments(segments)

    assert len(groups) == 1
    assert [segment.text for segment in groups[0]] == ["そう", "思うんですけど。"]


def test_build_batch_prompt_numbers_lines() -> None:
    prompt = _build_batch_prompt(["おはよう", "こんばんは"])

    assert "1. おはよう" in prompt
    assert "2. こんばんは" in prompt
    assert "按同样的编号输出 2 行" in prompt


def test_parse_batch_output_reads_numbered_lines() -> None:
    raw = "1. 早上好\n2. 晚上好"

    assert _parse_batch_output(raw, 2) == ["早上好", "晚上好"]


def test_parse_batch_output_falls_back_to_plain_lines() -> None:
    assert _parse_batch_output("早上好\n晚上好", 2) == ["早上好", "晚上好"]


def test_parse_batch_output_rejects_line_count_mismatch() -> None:
    assert _parse_batch_output("1. 早上好", 2) is None
    assert _parse_batch_output("早上好", 2) is None


def test_missing_glossary_terms_only_flags_source_hits() -> None:
    glossary = [GlossaryEntry(source="ヒカル", target="小光"), GlossaryEntry(source="みえ", target="美惠")]

    assert _missing_glossary_terms("小光来了", "ヒカルが来た", glossary) == []
    missing = _missing_glossary_terms("希卡尔来了", "ヒカルが来た", glossary)
    assert [entry.target for entry in missing] == ["小光"]
    # 原文没出现该词时不要求译文出现。
    assert _missing_glossary_terms("今天很冷", "今日は寒い", glossary) == []


class _ScriptedLlm:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict]] = []

    def create_chat_completion(self, messages, **_kwargs):
        self.calls.append(list(messages))
        text = self.replies.pop(0) if self.replies else ""
        return {"choices": [{"message": {"content": text}}]}


def test_translate_one_retries_when_glossary_term_is_missing() -> None:
    llm = _ScriptedLlm(["希卡尔来了", "小光来了"])

    result = _translate_one(
        llm,
        system_prompt="s",
        source_text="ヒカルが来た",
        history=[],
        glossary=[GlossaryEntry(source="ヒカル", target="小光")],
    )

    assert result == "小光来了"
    assert len(llm.calls) == 2
    # 第二次请求带着明确的术语要求，而不是只靠 system prompt。
    assert "ヒカル→小光" in llm.calls[1][-1]["content"]


def test_translate_one_keeps_usable_text_when_term_still_missing() -> None:
    llm = _ScriptedLlm(["希卡尔来了", "希卡尔来了"])

    result = _translate_one(
        llm,
        system_prompt="s",
        source_text="ヒカルが来た",
        history=[],
        glossary=[GlossaryEntry(source="ヒカル", target="小光")],
    )

    # 宁可译名不统一，也不要整条丢掉。
    assert result == "希卡尔来了"


def test_translate_batch_falls_back_when_line_count_mismatch() -> None:
    llm = _ScriptedLlm(["1. 早上好"])

    assert _translate_batch(
        llm,
        system_prompt="s",
        source_texts=["おはよう", "こんばんは"],
        history=[],
        glossary=[],
    ) == ["", ""]
