"""Expose pip nvidia-* CUDA libs to JamePeng llama-cpp-python.

Must run before anything imports llama_cpp / loads ggml backends. ggml registers
CUDA backends only at first load; missing cudart/cublas then means permanent CPU
for the process even if libs appear later.

Linux: LD_LIBRARY_PATH + symlink into llama_cpp/lib + RTLD_GLOBAL preload.
Windows: os.add_dll_directory / PATH for nvidia/*/bin|lib + ggml-cuda.dll probe.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger("ComfyUI-llama-cpp_vlm_fork.cuda")

_LINUX_CUDA_LIBS = (
    ("libnccl.so.2", "libnccl.so.2*"),
    ("libcudart.so.12", "libcudart.so.12*"),
    ("libcublas.so.12", "libcublas.so.12*"),
    ("libcublasLt.so.12", "libcublasLt.so.12*"),
)

# Windows JamePeng wheels in requirements.txt are +cu130.
_WIN_CUDA_DLLS = (
    "cudart64_13.dll",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
)


def _msg(level: str, text: str, *, use_print: bool) -> None:
    if use_print:
        print(f"[ComfyUI-llama-cpp_vlm_fork] {text}", flush=True)
    elif level == "warning":
        _log.warning("%s", text)
    else:
        _log.info("%s", text)


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


def _nvidia_dir_priority(path: Path) -> tuple[int, str]:
    parts = {p.lower() for p in path.parts}
    if "nccl" in parts:
        return (0, str(path))
    if "cuda_runtime" in parts:
        return (1, str(path))
    if "cublas" in parts:
        return (2, str(path))
    return (3, str(path))


def nvidia_lib_dirs() -> list[Path]:
    """Linux: site-packages/nvidia/*/lib with .so files."""
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
    dirs.sort(key=_nvidia_dir_priority)
    return dirs


def nvidia_dll_dirs() -> list[Path]:
    """Windows: site-packages/nvidia/*/bin and */lib containing .dll."""
    dirs: list[Path] = []
    seen: set[str] = set()
    for root in _site_package_roots():
        if not root.is_dir():
            continue
        for d in sorted(list(root.glob("nvidia/*/bin")) + list(root.glob("nvidia/*/lib"))):
            if not d.is_dir() or not any(d.glob("*.dll")):
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            dirs.append(d)
    dirs.sort(key=_nvidia_dir_priority)
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


def find_nvidia_dll(name: str) -> Path | None:
    for d in nvidia_dll_dirs():
        candidate = d / name
        if candidate.is_file():
            return candidate
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


def ggml_cuda_backend_path() -> Path | None:
    lib_dir = llama_cpp_lib_dir()
    if lib_dir is None:
        return None
    if sys.platform.startswith("linux"):
        p = lib_dir / "libggml-cuda.so"
        return p if p.is_file() else None
    if sys.platform == "win32":
        p = lib_dir / "ggml-cuda.dll"
        return p if p.is_file() else None
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


def _prepend_path(dirs: list[Path]) -> None:
    if not dirs:
        return
    extra = [str(d) for d in dirs]
    cur = os.environ.get("PATH", "")
    parts = extra + ([p for p in cur.split(os.pathsep) if p] if cur else [])
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    os.environ["PATH"] = os.pathsep.join(out)


def _add_dll_directories(dirs: list[Path]) -> None:
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    for d in dirs:
        try:
            add(str(d))
        except (OSError, FileNotFoundError):
            pass


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


def gpu_offload_hint() -> str:
    """Short install/verify hint for the current OS."""
    py = sys.executable
    if sys.platform.startswith("linux"):
        return (
            f"Install CUDA runtime into ComfyUI's venv and restart:\n"
            f"  {py} -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-nccl-cu12\n"
            f"Then verify: ldd .../llama_cpp/lib/libggml-cuda.so | grep -E 'cudart|cublas|not found'"
        )
    if sys.platform == "win32":
        return (
            f"Windows (+cu130 wheel): install CUDA 13 runtime into ComfyUI's venv and restart:\n"
            f"  {py} -m pip install nvidia-cuda-runtime-cu13 nvidia-cublas-cu13\n"
            f"Also confirm ggml-cuda.dll exists under .../llama_cpp/lib "
            f"(reinstall from this node's requirements.txt if missing)."
        )
    return "This platform has no CUDA GPU path in this fork (use Metal wheel on macOS)."


def ensure_cuda_libs_for_llama_cpp(*, use_print: bool = False) -> list[str]:
    """Make CUDA runtime libs visible to llama_cpp. Returns missing library names."""
    if sys.platform.startswith("linux"):
        return _ensure_linux(use_print=use_print)
    if sys.platform == "win32":
        return _ensure_windows(use_print=use_print)
    return []


def _ensure_linux(*, use_print: bool) -> list[str]:
    lib_dirs = nvidia_lib_dirs()
    _prepend_ld_library_path(lib_dirs)

    missing: list[str] = []
    for soname, pattern in _LINUX_CUDA_LIBS:
        src = find_nvidia_lib(soname, pattern)
        if src is None:
            missing.append(soname)
            continue
        linked = _link_into_llama_cpp(src, soname)
        if linked:
            _msg("info", f"llama-cpp: linked {linked} -> {src}", use_print=use_print)
        else:
            _msg(
                "warning",
                f"llama-cpp: found {soname} at {src} but could not link into llama_cpp/lib",
                use_print=use_print,
            )

    if missing:
        _msg(
            "warning",
            "llama-cpp: missing CUDA libs "
            + ", ".join(missing)
            + ". "
            + gpu_offload_hint().replace("\n", " "),
            use_print=use_print,
        )
    return missing


def _ensure_windows(*, use_print: bool) -> list[str]:
    dll_dirs = nvidia_dll_dirs()
    _add_dll_directories(dll_dirs)
    _prepend_path(dll_dirs)
    lib_dir = llama_cpp_lib_dir()
    if lib_dir is not None:
        _add_dll_directories([lib_dir])
        _prepend_path([lib_dir])

    for d in dll_dirs:
        _msg("info", f"llama-cpp: Windows DLL search path += {d}", use_print=use_print)

    missing = [name for name in _WIN_CUDA_DLLS if find_nvidia_dll(name) is None]
    if missing:
        _msg(
            "warning",
            "llama-cpp: missing CUDA DLLs "
            + ", ".join(missing)
            + ". "
            + gpu_offload_hint().replace("\n", " "),
            use_print=use_print,
        )
    else:
        for name in _WIN_CUDA_DLLS:
            src = find_nvidia_dll(name)
            if src:
                _msg("info", f"llama-cpp: found {src}", use_print=use_print)
    return missing


def preload_and_probe_ggml_cuda(*, use_print: bool = False) -> bool:
    """Preload CUDA runtime and probe ggml CUDA backend. True if backend loads."""
    if sys.platform.startswith("linux"):
        return _probe_linux(use_print=use_print)
    if sys.platform == "win32":
        return _probe_windows(use_print=use_print)
    return False


def _probe_linux(*, use_print: bool) -> bool:
    try:
        import ctypes

        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        for soname, pattern in _LINUX_CUDA_LIBS:
            src = find_nvidia_lib(soname, pattern)
            if src is not None:
                ctypes.CDLL(str(src), mode=mode)

        cuda_so = ggml_cuda_backend_path()
        if cuda_so is None:
            _msg("warning", "llama-cpp: libggml-cuda.so not found (CPU wheel?)", use_print=use_print)
            return False
        try:
            ctypes.CDLL(str(cuda_so), mode=mode)
            _msg("info", "llama-cpp: libggml-cuda.so load OK", use_print=use_print)
            return True
        except OSError as e:
            _msg(
                "warning",
                f"llama-cpp: libggml-cuda.so failed to load ({e}). {gpu_offload_hint()}",
                use_print=use_print,
            )
            return False
    except OSError as e:
        _msg("warning", f"CUDA lib preload skipped: {e}", use_print=use_print)
        return False


def _probe_windows(*, use_print: bool) -> bool:
    try:
        import ctypes

        for name in _WIN_CUDA_DLLS:
            src = find_nvidia_dll(name)
            if src is not None:
                try:
                    ctypes.WinDLL(str(src))
                except OSError as e:
                    _msg(
                        "warning",
                        f"llama-cpp: failed to load {src} ({e})",
                        use_print=use_print,
                    )

        cuda_dll = ggml_cuda_backend_path()
        if cuda_dll is None:
            _msg(
                "warning",
                "llama-cpp: ggml-cuda.dll not found under llama_cpp/lib "
                "(CPU wheel or wrong package). Reinstall from requirements.txt.",
                use_print=use_print,
            )
            return False
        try:
            ctypes.WinDLL(str(cuda_dll))
            _msg("info", "llama-cpp: ggml-cuda.dll load OK", use_print=use_print)
            return True
        except OSError as e:
            _msg(
                "warning",
                f"llama-cpp: ggml-cuda.dll failed to load ({e}). {gpu_offload_hint()}",
                use_print=use_print,
            )
            return False
    except OSError as e:
        _msg("warning", f"CUDA DLL preload skipped: {e}", use_print=use_print)
        return False
