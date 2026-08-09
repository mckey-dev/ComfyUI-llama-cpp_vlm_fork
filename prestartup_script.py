"""Install llama-cpp-python for this node before ComfyUI imports it.

ComfyUI Manager only reads pyproject.toml, which cannot list OS/Python-specific
wheel URLs. requirements.txt is the source of truth; this script installs the
matching llama-cpp-python line when the package is absent.

Failures are logged (not raised) so other custom nodes keep loading.
A real import check of llama_cpp belongs in __init__.py after NCCL setup.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

REQ_PATH = Path(__file__).resolve().parent / "requirements.txt"
_PREFIX = "[ComfyUI-llama-cpp_vlm_fork]"


def _package_present(name: str) -> bool:
    """True if the distribution/module can be located (not a native-load test)."""
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _python_version_tag() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _marker_matches(markers: str, py: str, system: str) -> bool:
    """Evaluate environment markers; fall back to simple substring checks."""
    markers = markers.strip()
    try:
        from packaging.markers import Marker

        return bool(Marker(markers).evaluate())
    except Exception:
        pass

    # Fallback for environments without packaging, or odd marker strings.
    normalized = markers.replace("'", '"').replace(" ", "")
    need_py = f'python_version=="{py}"'
    need_os = f'platform_system=="{system}"'
    return need_py in normalized and need_os in normalized


def _wheel_url_from_requirements(req_path: Path) -> str | None:
    """Pick llama-cpp-python @ URL matching current Python + OS markers."""
    if not req_path.is_file():
        return None

    py = _python_version_tag()
    system = platform.system()  # Windows / Linux / Darwin

    for raw in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.lower().startswith("llama-cpp-python"):
            continue
        if " @ " not in line or ";" not in line:
            continue
        pkg_url, markers = line.split(";", 1)
        if not _marker_matches(markers, py, system):
            continue
        return pkg_url.split("@", 1)[1].strip()
    return None


def _pip_install(target: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", target]
    print(f"{_PREFIX} + {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def ensure_llama_cpp_python() -> None:
    if _package_present("llama_cpp"):
        print(f"{_PREFIX} llama-cpp-python: already present", flush=True)
        return

    print(
        f"{_PREFIX} llama-cpp-python not found; installing from requirements.txt…",
        flush=True,
    )
    wheel = _wheel_url_from_requirements(REQ_PATH)
    if not wheel:
        py = _python_version_tag()
        system = platform.system()
        print(
            f"{_PREFIX} [warn] no matching llama-cpp-python wheel in requirements.txt "
            f"(python={py}, system={system}). Update {REQ_PATH}",
            flush=True,
        )
        return

    try:
        _pip_install(wheel)
    except subprocess.CalledProcessError as e:
        print(
            f"{_PREFIX} [warn] pip install failed: {e}\n"
            f"  manual: {sys.executable} -m pip install -r \"{REQ_PATH}\"",
            flush=True,
        )
        return

    if not _package_present("llama_cpp"):
        print(
            f"{_PREFIX} [warn] install finished but llama_cpp is still missing. "
            f"Try: {sys.executable} -m pip install -r \"{REQ_PATH}\"",
            flush=True,
        )
        return

    print(f"{_PREFIX} llama-cpp-python: install complete", flush=True)


try:
    ensure_llama_cpp_python()
except Exception as e:
    print(f"{_PREFIX} [warn] prestartup failed: {e}", flush=True)
