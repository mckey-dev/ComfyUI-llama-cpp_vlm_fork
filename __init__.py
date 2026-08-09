"""ComfyUI-llama-cpp_vlm_fork entrypoint.

JamePeng CUDA wheels often link against libnccl.so.2 but do not bundle it.
Resolve NCCL from the torch-shipped nvidia-nccl package before importing nodes
(which import llama_cpp).

Changing LD_LIBRARY_PATH after process start is unreliable for dlopen, so we
prefer placing a symlink next to libggml.so (llama_cpp/lib) when $ORIGIN is used.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)


def _site_package_roots() -> list[Path]:
    import site

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
    # dedupe preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _nvidia_lib_dirs() -> list[Path]:
    """Return nvidia/*/lib directories under site-packages (nccl first)."""
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in _site_package_roots():
        if not root.is_dir():
            continue
        for d in sorted(root.glob("nvidia/*/lib")):
            if not d.is_dir() or not any(d.glob("*.so*")):
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            dirs.append(d)
    dirs.sort(key=lambda p: (0 if "nccl" in p.parts else 1, str(p)))
    return dirs


def _find_libnccl() -> Path | None:
    for d in _nvidia_lib_dirs():
        candidate = d / "libnccl.so.2"
        if candidate.exists():
            return candidate
        matches = sorted(d.glob("libnccl.so.2*"))
        if matches:
            return matches[0]
    return None


def _llama_cpp_lib_dir() -> Path | None:
    """Locate llama_cpp/lib without importing the native extension."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("llama_cpp")
        if spec is not None:
            if spec.submodule_search_locations:
                lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
                if lib_dir.is_dir():
                    return lib_dir
            if spec.origin:
                lib_dir = Path(spec.origin).resolve().parent / "lib"
                if lib_dir.is_dir():
                    return lib_dir
    except Exception:
        pass
    for root in _site_package_roots():
        lib_dir = root / "llama_cpp" / "lib"
        if lib_dir.is_dir():
            return lib_dir
    return None


def _prepend_ld_library_path(dirs: list[Path]) -> None:
    if not dirs:
        return
    extra = [str(d) for d in dirs]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    parts = extra + ([p for p in cur.split(":") if p] if cur else [])
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(out)


def _link_nccl_into_llama_cpp(nccl: Path) -> Path | None:
    lib_dir = _llama_cpp_lib_dir()
    if lib_dir is None:
        return None
    dest = lib_dir / "libnccl.so.2"
    try:
        if dest.exists() or dest.is_symlink():
            try:
                if dest.resolve() == nccl.resolve():
                    return dest
            except OSError:
                pass
            dest.unlink()
        dest.symlink_to(nccl.resolve())
        return dest
    except OSError as e:
        _log.warning("Could not symlink libnccl into llama_cpp/lib: %s", e)
        return None


def _ensure_nccl_for_llama_cpp() -> None:
    """Make libnccl.so.2 visible to JamePeng CUDA llama-cpp-python wheels (Linux)."""
    if not sys.platform.startswith("linux"):
        return

    lib_dirs = _nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    nccl = _find_libnccl()
    if nccl is None:
        _log.warning(
            "libnccl.so.2 not found under site-packages/nvidia/nccl/lib. "
            "CUDA llama-cpp-python may fail to import; ensure torch's nvidia-nccl "
            "package is installed (e.g. nvidia-nccl-cu12 / nvidia-nccl-cu13)."
        )
        return

    linked = _link_nccl_into_llama_cpp(nccl)
    if linked:
        _log.info("llama-cpp: linked NCCL %s -> %s", linked, nccl)
    else:
        _log.warning(
            "llama-cpp: libnccl found at %s but llama_cpp/lib was missing; "
            "set LD_LIBRARY_PATH to include nvidia/nccl/lib before starting ComfyUI "
            "if import fails.",
            nccl,
        )


_ensure_nccl_for_llama_cpp()

# Prefer RTLD_GLOBAL preload if symlink alone is not enough (Linux).
if sys.platform.startswith("linux"):
    try:
        import ctypes

        nccl = _find_libnccl()
        if nccl is not None:
            ctypes.CDLL(str(nccl), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError as e:
        _log.debug("NCCL preload skipped: %s", e)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    _log.error(
        "Failed to load ComfyUI-llama-cpp_vlm_fork nodes (llama-cpp-python / NCCL / CUDA). "
        "Install with: python -m pip install -r \"%s\" — %s",
        Path(__file__).resolve().parent / "requirements.txt",
        e,
    )
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]