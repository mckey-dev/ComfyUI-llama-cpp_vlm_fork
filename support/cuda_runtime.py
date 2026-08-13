"""Expose pip nvidia-* CUDA libs to JamePeng llama-cpp-python (Linux).

Must run before anything imports llama_cpp / dlopens libggml*.so. ggml registers
CUDA backends only at first load; missing libcudart then means permanent CPU
for the process even if libs appear later.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("ComfyUI-llama-cpp_vlm_fork.cuda")

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


def nvidia_lib_dirs() -> list[Path]:
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


def find_nvidia_lib(soname: str, pattern: str) -> Path | None:
    for d in nvidia_lib_dirs():
        candidate = d / soname
        if candidate.exists():
            return candidate
        matches = sorted(d.glob(pattern))
        if matches:
            return matches[0]
    return None


def llama_cpp_lib_dir() -> Path | None:
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
    lib_dir = llama_cpp_lib_dir()
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


def ensure_cuda_libs_for_llama_cpp(*, use_print: bool = False) -> list[str]:
    """Symlink CUDA libs into llama_cpp/lib and prepend LD_LIBRARY_PATH.

    Returns list of missing sonames. No-op on non-Linux.
    """
    if not sys.platform.startswith("linux"):
        return []

    lib_dirs = nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    def _msg(level: str, text: str) -> None:
        if use_print:
            prefix = "[ComfyUI-llama-cpp_vlm_fork]"
            print(f"{prefix} {text}", flush=True)
        elif level == "warning":
            _log.warning("%s", text)
        else:
            _log.info("%s", text)

    missing: list[str] = []
    for soname, pattern in _CUDA_RUNTIME_LIBS:
        src = find_nvidia_lib(soname, pattern)
        if src is None:
            missing.append(soname)
            continue
        linked = _link_into_llama_cpp(src, soname)
        if linked:
            _msg("info", f"llama-cpp: linked {linked} -> {src}")
        else:
            _msg(
                "warning",
                f"llama-cpp: found {soname} at {src} but could not link into llama_cpp/lib",
            )

    if missing:
        _msg(
            "warning",
            "llama-cpp: missing CUDA libs "
            + ", ".join(missing)
            + " under site-packages/nvidia/*/lib. "
            "GPU offload will fall back to CPU. Install e.g. "
            "nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nccl-cu12",
        )
    return missing


def preload_and_probe_ggml_cuda(*, use_print: bool = False) -> bool:
    """RTLD_GLOBAL preload + early libggml-cuda.so probe. Returns True if CUDA .so loads."""
    if not sys.platform.startswith("linux"):
        return False

    def _msg(level: str, text: str) -> None:
        if use_print:
            print(f"[ComfyUI-llama-cpp_vlm_fork] {text}", flush=True)
        elif level == "warning":
            _log.warning("%s", text)
        else:
            _log.info("%s", text)

    try:
        import ctypes

        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        for soname, pattern in _CUDA_RUNTIME_LIBS:
            src = find_nvidia_lib(soname, pattern)
            if src is not None:
                ctypes.CDLL(str(src), mode=mode)

        lib_dir = llama_cpp_lib_dir()
        cuda_so = (lib_dir / "libggml-cuda.so") if lib_dir else None
        if cuda_so is None or not cuda_so.is_file():
            _msg("warning", "llama-cpp: libggml-cuda.so not found (CPU wheel?)")
            return False
        try:
            ctypes.CDLL(str(cuda_so), mode=mode)
            _msg("info", "llama-cpp: libggml-cuda.so load OK")
            return True
        except OSError as e:
            _msg(
                "warning",
                f"llama-cpp: libggml-cuda.so failed to load ({e}). "
                "GPU offload will fall back to CPU. Install: "
                "python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12",
            )
            return False
    except OSError as e:
        _msg("warning", f"CUDA lib preload skipped: {e}")
        return False
