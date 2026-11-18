"""Local subtitle generation CLI."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
USER_HF_HOME = Path.home() / ".cache" / "huggingface"

_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def _hf_home_has_models(path: Path) -> bool:
    """Return True when the huggingface home already contains downloaded models."""
    hub = path / "hub"
    if not hub.is_dir():
        return False
    try:
        return any(child.name.startswith("models--") for child in hub.iterdir())
    except OSError:  # pragma: no cover - 无权限时的保守回退
        return False


def resolve_hf_home(
    project_home: Path | None = None,
    user_home: Path | None = None,
) -> Path:
    """Pick the HuggingFace cache directory that actually holds models.

    优先级：``SUBGEN_HF_HOME`` 环境变量 > 含模型的项目内缓存 > 含模型的用户缓存 >
    项目内缓存（待创建）。这样既保留"缓存跟着项目走"的可移植性，又不会在用户
    已经下载过模型时重新下载一份。
    """
    explicit = os.environ.get("SUBGEN_HF_HOME")
    if explicit:
        return Path(explicit).expanduser()

    project = (project_home or DEFAULT_HF_HOME).expanduser()
    user = (user_home or USER_HF_HOME).expanduser()

    if _hf_home_has_models(project):
        return project
    if _hf_home_has_models(user):
        return user
    return project


def _configure_hf_home() -> None:
    """Set HF_HOME once, before faster-whisper is imported."""
    if os.environ.get("HF_HOME"):
        return
    os.environ["HF_HOME"] = str(resolve_hf_home())


def _windows_system_proxy() -> str | None:
    """Read the WinINET proxy (what VPN clients set in Windows settings)."""
    if sys.platform != "win32":  # pragma: no cover - 仅在 Windows 上有意义
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            server, _ = winreg.QueryValueEx(key, "ProxyServer")
    except OSError:  # pragma: no cover - 键不存在或无权访问
        return None
    if not enabled or not server:
        return None

    raw = str(server).strip()
    if not raw:
        return None
    # ProxyServer 既可能是 "host:port"，也可能是 "http=host:port;https=host:port"
    if "=" in raw:
        per_protocol = {}
        for item in raw.split(";"):
            name, _, value = item.partition("=")
            if name.strip() and value.strip():
                per_protocol[name.strip().lower()] = value.strip()
        raw = per_protocol.get("https") or per_protocol.get("http") or ""
        if not raw:
            return None
    if not raw.lower().startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw


def configure_proxy_from_system(
    env: MutableMapping[str, str] | None = None,
) -> str | None:
    """Apply the Windows system proxy to HTTP(S)_PROXY when nothing is configured.

    Python 的 requests/huggingface_hub 不会读取 Windows 的"Internet 选项"代理，
    所以开着 VPN 也常常下载不动模型。这里在导入依赖前把系统代理搬进环境变量。
    设置 ``SUBGEN_NO_SYSTEM_PROXY=1`` 可强制直连。
    """
    target = os.environ if env is None else env
    if target.get("SUBGEN_NO_SYSTEM_PROXY"):
        return None
    if any(target.get(key) for key in _PROXY_ENV_KEYS):
        return None
    proxy = _windows_system_proxy()
    if not proxy:
        return None
    for key in _PROXY_ENV_KEYS:
        target[key] = proxy
    return proxy


def _configure_environment() -> None:
    configure_proxy_from_system()
    _configure_hf_home()


_configure_environment()

__version__ = "0.5.0"
