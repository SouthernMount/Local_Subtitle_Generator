from pathlib import Path

import subgen
from subgen import translate


def test_translation_model_env_var_wins(monkeypatch) -> None:
    monkeypatch.setenv("SUBGEN_TRANSLATION_MODEL", "custom-model")

    assert translate.resolve_default_translation_model() == "custom-model"


def test_default_hf_home_points_into_project() -> None:
    assert subgen.DEFAULT_HF_HOME == subgen.PROJECT_ROOT / ".cache" / "huggingface"


def test_resolve_hf_home_prefers_project_cache_with_models(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SUBGEN_HF_HOME", raising=False)
    project = tmp_path / "project"
    user = tmp_path / "user"
    (project / "hub" / "models--a--b").mkdir(parents=True)
    (user / "hub" / "models--c--d").mkdir(parents=True)

    assert subgen.resolve_hf_home(project, user) == project


def test_resolve_hf_home_falls_back_to_user_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SUBGEN_HF_HOME", raising=False)
    project = tmp_path / "project"
    user = tmp_path / "user"
    (user / "hub" / "models--c--d").mkdir(parents=True)

    assert subgen.resolve_hf_home(project, user) == user


def test_resolve_hf_home_defaults_to_project_when_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SUBGEN_HF_HOME", raising=False)

    assert subgen.resolve_hf_home(tmp_path / "project", tmp_path / "user") == (
        tmp_path / "project"
    )


def test_resolve_hf_home_explicit_env_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUBGEN_HF_HOME", str(tmp_path / "custom"))

    assert subgen.resolve_hf_home(tmp_path / "project", tmp_path / "user") == (
        tmp_path / "custom"
    )


def test_configure_proxy_from_system_sets_all_env_keys(monkeypatch) -> None:
    monkeypatch.setattr(subgen, "_windows_system_proxy", lambda: "http://127.0.0.1:7890")
    env: dict[str, str] = {}

    assert subgen.configure_proxy_from_system(env) == "http://127.0.0.1:7890"
    assert env["HTTP_PROXY"] == env["HTTPS_PROXY"] == env["http_proxy"] == "http://127.0.0.1:7890"


def test_configure_proxy_keeps_existing_settings(monkeypatch) -> None:
    monkeypatch.setattr(subgen, "_windows_system_proxy", lambda: "http://127.0.0.1:7890")
    env = {"HTTPS_PROXY": "http://existing:1"}

    assert subgen.configure_proxy_from_system(env) is None
    assert env == {"HTTPS_PROXY": "http://existing:1"}


def test_configure_proxy_can_be_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setattr(subgen, "_windows_system_proxy", lambda: "http://127.0.0.1:7890")
    env = {"SUBGEN_NO_SYSTEM_PROXY": "1"}

    assert subgen.configure_proxy_from_system(env) is None
    assert "HTTPS_PROXY" not in env


def test_configure_proxy_noop_when_no_system_proxy(monkeypatch) -> None:
    monkeypatch.setattr(subgen, "_windows_system_proxy", lambda: None)
    env: dict[str, str] = {}

    assert subgen.configure_proxy_from_system(env) is None
    assert env == {}


def test_translation_model_defaults_to_project_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SUBGEN_TRANSLATION_MODEL", raising=False)
    monkeypatch.setattr(translate, "LOCAL_TRANSLATION_MODEL_DIR", tmp_path / "qwen")

    assert translate.resolve_default_translation_model() == str(tmp_path / "qwen")
