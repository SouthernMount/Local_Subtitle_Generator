from pathlib import Path

import pytest

from subgen import cli
from subgen.glossary import GlossaryEntry
from subgen.transcribe import TranscriptSegment, TranscriptionResult


@pytest.fixture()
def media_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.mp4"
    path.write_bytes(b"placeholder")
    return path


@pytest.fixture(autouse=True)
def fake_transcription_and_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_transcribe_file_with_language(*_args, **_kwargs):
        return TranscriptionResult(
            segments=[TranscriptSegment(0, 1.25, "Hello, world.")],
            language="en",
        )

    def fake_translate_segments_to_chinese(segments, **_kwargs):
        return [
            TranscriptSegment(segment.start, segment.end, "你好，世界。")
            for segment in segments
        ]

    monkeypatch.setattr(cli, "transcribe_file_with_language", fake_transcribe_file_with_language)
    monkeypatch.setattr(cli, "translate_segments_to_chinese", fake_translate_segments_to_chinese)


def test_cli_default_outputs_srt_and_vtt(media_file: Path) -> None:
    outputs = cli.generate_subtitles(media_file)

    assert outputs == [media_file.with_suffix(".srt"), media_file.with_suffix(".vtt")]
    assert media_file.with_suffix(".srt").read_text(encoding="utf-8").startswith("1\n")
    assert media_file.with_suffix(".vtt").read_text(encoding="utf-8").startswith("WEBVTT\n")


def test_cli_outputs_translated_chinese_text(media_file: Path) -> None:
    cli.generate_subtitles(media_file, subtitle_format=cli.SubtitleFormat.srt)

    assert "你好，世界。" in media_file.with_suffix(".srt").read_text(encoding="utf-8")
    assert "Hello, world." not in media_file.with_suffix(".srt").read_text(encoding="utf-8")


def test_cli_uses_auto_source_language_by_default(
    media_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_transcribe_file_with_language(*_args, **kwargs):
        calls.append(kwargs["language"])
        return TranscriptionResult([], language="en")

    monkeypatch.setattr(cli, "transcribe_file_with_language", fake_transcribe_file_with_language)

    cli.generate_subtitles(media_file)

    assert calls == ["auto"]


def test_cli_passes_compute_type_to_transcription(
    media_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_transcribe_file_with_language(*_args, **kwargs):
        calls.append(kwargs["compute_type"])
        return TranscriptionResult([], language="zh")

    monkeypatch.setattr(cli, "transcribe_file_with_language", fake_transcribe_file_with_language)

    cli.generate_subtitles(media_file, compute_type="float16")

    assert calls == ["float16"]


def test_cli_format_srt_only(media_file: Path) -> None:
    outputs = cli.generate_subtitles(media_file, subtitle_format=cli.SubtitleFormat.srt)

    assert outputs == [media_file.with_suffix(".srt")]
    assert media_file.with_suffix(".srt").exists()
    assert not media_file.with_suffix(".vtt").exists()


def test_cli_format_vtt_only(media_file: Path) -> None:
    outputs = cli.generate_subtitles(media_file, subtitle_format=cli.SubtitleFormat.vtt)

    assert outputs == [media_file.with_suffix(".vtt")]
    assert media_file.with_suffix(".vtt").exists()
    assert not media_file.with_suffix(".srt").exists()


def test_cli_custom_output_dir(media_file: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "subtitles"

    outputs = cli.generate_subtitles(media_file, output_dir=output_dir)

    assert outputs == [output_dir / "input.srt", output_dir / "input.vtt"]
    assert (output_dir / "input.srt").exists()
    assert (output_dir / "input.vtt").exists()


def test_cli_requires_overwrite_for_existing_files(media_file: Path) -> None:
    media_file.with_suffix(".srt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        cli.generate_subtitles(media_file, subtitle_format=cli.SubtitleFormat.srt)


def test_cli_overwrite_replaces_existing_files(media_file: Path) -> None:
    srt_path = media_file.with_suffix(".srt")
    srt_path.write_text("old", encoding="utf-8")

    cli.generate_subtitles(
        media_file,
        subtitle_format=cli.SubtitleFormat.srt,
        overwrite=True,
    )

    assert "你好，世界。" in srt_path.read_text(encoding="utf-8")


def test_cli_passes_glossary_and_model_path_to_translation(
    media_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_translate(segments, **kwargs):
        captured.update(kwargs)
        return [
            TranscriptSegment(segment.start, segment.end, "你好。")
            for segment in segments
        ]

    monkeypatch.setattr(cli, "translate_segments_to_chinese", fake_translate)

    cli.generate_subtitles(
        media_file,
        subtitle_format=cli.SubtitleFormat.srt,
        glossary=[GlossaryEntry("ヒカル", "光")],
    )

    assert captured["glossary"] == [GlossaryEntry("ヒカル", "光")]
    assert captured["model_path"] == cli.DEFAULT_TRANSLATION_MODEL


def test_cli_applies_chinese_rename(media_file: Path) -> None:
    cli.generate_subtitles(
        media_file,
        subtitle_format=cli.SubtitleFormat.srt,
        rename="你好=您好",
    )

    text = media_file.with_suffix(".srt").read_text(encoding="utf-8")
    assert "您好，世界。" in text
    assert "你好，世界。" not in text


def test_cli_missing_input_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Input file does not exist"):
        cli.generate_subtitles(tmp_path / "missing.mp4")


def test_cli_debug_outputs_are_off_by_default(media_file: Path) -> None:
    cli.generate_subtitles(media_file, subtitle_format=cli.SubtitleFormat.srt)

    assert not (media_file.parent / "debug").exists()


def test_cli_debug_originals_writes_diagnostic_files(media_file: Path) -> None:
    cli.generate_subtitles(
        media_file,
        subtitle_format=cli.SubtitleFormat.srt,
        debug_originals=True,
    )

    debug_dir = media_file.parent / "debug"
    assert (debug_dir / "input.ja.srt").exists()
    assert (debug_dir / "input.bilingual.srt").exists()
    qa_lines = (debug_dir / "input.qa.jsonl").read_text(encoding="utf-8").splitlines()
    assert '"meta"' in qa_lines[0]
    assert '"source"' in qa_lines[1]


def test_download_model_main_invokes_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    whisper_calls = []
    verify_calls = []

    def fake_download_model(*, model_name: str, device: str, compute_type: str) -> None:
        whisper_calls.append((model_name, device, compute_type))

    def fake_verify_translation_model(model_path: str) -> Path:
        verify_calls.append(model_path)
        return Path(model_path) / "model.gguf"

    monkeypatch.setattr(cli, "download_model", fake_download_model)
    monkeypatch.setattr(cli, "verify_translation_model", fake_verify_translation_model)
    monkeypatch.setattr("sys.argv", ["subgen-download-model", "--model", "base", "--device", "cpu"])

    cli.download_model_main()

    assert whisper_calls == [("base", "cpu", "default")]
    assert verify_calls == [cli.DEFAULT_TRANSLATION_MODEL]


def test_download_model_main_can_skip_translation_model(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_calls = []

    def fake_download_model(*, model_name: str, device: str, compute_type: str) -> None:
        return None

    def fake_verify_translation_model(model_path: str) -> Path:
        verify_calls.append(model_path)
        return Path(model_path)

    monkeypatch.setattr(cli, "download_model", fake_download_model)
    monkeypatch.setattr(cli, "verify_translation_model", fake_verify_translation_model)
    monkeypatch.setattr("sys.argv", ["subgen-download-model", "--skip-translation-model"])

    cli.download_model_main()

    assert verify_calls == []
