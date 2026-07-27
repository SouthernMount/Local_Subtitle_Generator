import importlib
import importlib.util
import os
from pathlib import Path

import subgen
from subgen import translate


def test_translation_model_env_var_wins(monkeypatch) -> None:
    monkeypatch.setenv("SUBGEN_TRANSLATION_MODEL", "custom-model")

    assert translate.resolve_default_translation_model() == "custom-model"


def test_package_sets_project_local_hf_cache(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    importlib.reload(subgen)

    assert os.environ["HF_HOME"] == str(subgen.DEFAULT_HF_HOME)
    assert subgen.DEFAULT_HF_HOME == subgen.PROJECT_ROOT / ".cache" / "huggingface"


def test_translation_model_uses_local_dir(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "models" / "nllb-200-distilled-600M"
    model_dir.mkdir(parents=True)
    monkeypatch.delenv("SUBGEN_TRANSLATION_MODEL", raising=False)
    monkeypatch.setattr(translate, "LOCAL_TRANSLATION_MODEL_DIR", model_dir)

    assert translate.resolve_default_translation_model() == str(model_dir)


def test_translation_model_falls_back_to_hf_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SUBGEN_TRANSLATION_MODEL", raising=False)
    monkeypatch.setattr(translate, "LOCAL_TRANSLATION_MODEL_DIR", tmp_path / "missing")

    assert translate.resolve_default_translation_model() == translate.HF_TRANSLATION_MODEL_ID


def test_download_script_detects_complete_model_dir(tmp_path: Path) -> None:
    module = _load_download_script()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for filename in (
        "config.json",
        "generation_config.json",
        "sentencepiece.bpe.model",
        "pytorch_model.bin",
        "tokenizer_config.json",
    ):
        (model_dir / filename).write_text("{}", encoding="utf-8")

    assert module._model_dir_is_complete(model_dir) is True


def test_download_script_rejects_partial_model_dir(tmp_path: Path) -> None:
    module = _load_download_script()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    assert module._model_dir_is_complete(model_dir) is False


def _load_download_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.py"
    spec = importlib.util.spec_from_file_location("download_models", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
