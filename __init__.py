"""ComfyUI-llama-cpp_vlm_fork entrypoint.

JamePeng CUDA wheels link against libnccl / libcudart / libcublas but do not
bundle them. Resolve those from pip nvidia-* packages before importing nodes
(which import llama_cpp and later dlopen libggml-cuda.so).

Changing LD_LIBRARY_PATH after process start is unreliable for dlopen, so we
also symlink next to libggml.so (llama_cpp/lib) when $ORIGIN is used.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

# soname -> glob under nvidia/*/lib (cu12 wheels used by JamePeng cu128 builds)
_CUDA_RUNTIME_LIBS = (
    ("libnccl.so.2", "libnccl.so.2*"),
    ("libcudart.so.12", "libcudart.so.12*"),
    ("libcublas.so.12", "libcublas.so.12*"),
    ("libcublasLt.so.12", "libcublasLt.so.12*"),
)


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


def _nvidia_lib_dir_priority(path: Path) -> tuple[int, str]:
    parts = {p.lower() for p in path.parts}
    if "nccl" in parts:
        return (0, str(path))
    if "cuda_runtime" in parts:
        return (1, str(path))
    if "cublas" in parts:
        return (2, str(path))
    return (3, str(path))


def _nvidia_lib_dirs() -> list[Path]:
    """Return nvidia/*/lib directories under site-packages (nccl / cudart / cublas first)."""
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
    dirs.sort(key=_nvidia_lib_dir_priority)
    return dirs


def _find_nvidia_lib(soname: str, pattern: str) -> Path | None:
    for d in _nvidia_lib_dirs():
        candidate = d / soname
        if candidate.exists():
            return candidate
        matches = sorted(d.glob(pattern))
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


def _link_into_llama_cpp(src: Path, soname: str) -> Path | None:
    lib_dir = _llama_cpp_lib_dir()
    if lib_dir is None:
        return None
    dest = lib_dir / soname
    try:
        if dest.exists() or dest.is_symlink():
            try:
                if dest.resolve() == src.resolve():
                    return dest
            except OSError:
                pass
            dest.unlink()
        dest.symlink_to(src.resolve())
        return dest
    except OSError as e:
        _log.warning("Could not symlink %s into llama_cpp/lib: %s", soname, e)
        return None


def _ensure_cuda_libs_for_llama_cpp() -> None:
    """Expose NCCL / CUDA runtime libs to JamePeng CUDA llama-cpp-python (Linux)."""
    if not sys.platform.startswith("linux"):
        return

    lib_dirs = _nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    missing: list[str] = []
    for soname, pattern in _CUDA_RUNTIME_LIBS:
        src = _find_nvidia_lib(soname, pattern)
        if src is None:
            missing.append(soname)
            continue
        linked = _link_into_llama_cpp(src, soname)
        if linked:
            _log.info("llama-cpp: linked %s -> %s", linked, src)
        else:
            _log.warning(
                "llama-cpp: found %s at %s but could not link into llama_cpp/lib",
                soname,
                src,
            )

    if missing:
        _log.warning(
            "llama-cpp: missing CUDA libs %s under site-packages/nvidia/*/lib. "
            "GPU offload will fall back to CPU. Install e.g. "
            "nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nccl-cu12 "
            "(or the cu13 variants matching your torch).",
            ", ".join(missing),
        )


_ensure_cuda_libs_for_llama_cpp()

# Prefer RTLD_GLOBAL preload if symlink alone is not enough (Linux).
if sys.platform.startswith("linux"):
    try:
        import ctypes

        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        for soname, pattern in _CUDA_RUNTIME_LIBS:
            src = _find_nvidia_lib(soname, pattern)
            if src is not None:
                ctypes.CDLL(str(src), mode=mode)
    except OSError as e:
        _log.debug("CUDA lib preload skipped: %s", e)

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