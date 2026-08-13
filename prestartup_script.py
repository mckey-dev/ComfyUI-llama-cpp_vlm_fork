"""Install runtime deps for this node before ComfyUI imports it.

ComfyUI Manager only reads pyproject.toml, which cannot list OS/Python-specific
wheel URLs. requirements.txt is the source of truth for llama-cpp-python.

On Linux, JamePeng CUDA wheels also need libcudart/libcublas (and usually
libnccl). Those are pulled from pip nvidia-*-cu12 packages when missing.

Failures are logged (not raised) so other custom nodes keep loading.
Native load / symlink setup belongs in __init__.py after this script runs.
"""

from __future__ import annotations

import platform
import site
import subprocess
import sys
from pathlib import Path

REQ_PATH = Path(__file__).resolve().parent / "requirements.txt"
_PREFIX = "[ComfyUI-llama-cpp_vlm_fork]"

# Linux CUDA runtime libs required by libggml-cuda.so (JamePeng cu12x wheels).
_LINUX_CUDA_PIP = (
    ("nvidia-cuda-runtime-cu12", "cuda_runtime", "libcudart.so.12"),
    ("nvidia-cublas-cu12", "cublas", "libcublas.so.12"),
    ("nvidia-nccl-cu12", "nccl", "libnccl.so.2"),
)


def _package_present(name: str) -> bool:
    """True if the distribution/module can be located (not a native-load test)."""
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _site_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        us = site.getusersitepackages()
        if us:
            roots.append(Path(us))
    except Exception:
        pass
    for p in sys.path:
        if p:
            roots.append(Path(p))
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _nvidia_so_present(subdir: str, soname: str) -> bool:
    """True if site-packages/nvidia/<subdir>/lib/<soname>* exists."""
    for root in _site_roots():
        lib = root / "nvidia" / subdir / "lib"
        if not lib.is_dir():
            continue
        if (lib / soname).exists():
            return True
        if any(lib.glob(soname + "*")):
            return True
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


def _pip_install(*targets: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *targets]
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


def ensure_linux_cuda_runtime() -> None:
    """Install pip nvidia-* packages needed by libggml-cuda.so on Linux."""
    if not sys.platform.startswith("linux"):
        return

    missing_pkgs: list[str] = []
    for pip_name, subdir, soname in _LINUX_CUDA_PIP:
        if _nvidia_so_present(subdir, soname):
            print(f"{_PREFIX} CUDA lib OK: {soname}", flush=True)
        else:
            print(f"{_PREFIX} CUDA lib missing: {soname} -> will install {pip_name}", flush=True)
            missing_pkgs.append(pip_name)

    if not missing_pkgs:
        return

    # Dedupe while preserving order
    to_install = list(dict.fromkeys(missing_pkgs))
    print(
        f"{_PREFIX} installing Linux CUDA runtime packages: {', '.join(to_install)}",
        flush=True,
    )
    try:
        _pip_install(*to_install)
    except subprocess.CalledProcessError as e:
        print(
            f"{_PREFIX} [warn] CUDA runtime pip install failed: {e}\n"
            f"  manual: {sys.executable} -m pip install {' '.join(to_install)}\n"
            f"  Without these, llama.cpp GPU offload falls back to CPU.",
            flush=True,
        )
        return

    still_missing = [
        soname
        for _, subdir, soname in _LINUX_CUDA_PIP
        if not _nvidia_so_present(subdir, soname)
    ]
    if still_missing:
        print(
            f"{_PREFIX} [warn] still missing after install: {', '.join(still_missing)}",
            flush=True,
        )
    else:
        print(f"{_PREFIX} Linux CUDA runtime packages: install complete", flush=True)


try:
    ensure_llama_cpp_python()
    ensure_linux_cuda_runtime()
except Exception as e:
    print(f"{_PREFIX} [warn] prestartup failed: {e}", flush=True)
