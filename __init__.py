"""ComfyUI-llama-cpp_vlm_fork entrypoint.

CUDA lib exposure for Linux is primarily done in prestartup_script.py (before
any custom node can import llama_cpp). This module re-runs the same setup as a
safety net, then imports nodes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

if sys.platform.startswith("linux"):
    try:
        from .support.cuda_runtime import (
            ensure_cuda_libs_for_llama_cpp,
            preload_and_probe_ggml_cuda,
        )

        ensure_cuda_libs_for_llama_cpp()
        preload_and_probe_ggml_cuda()
    except Exception as e:
        _log.warning("CUDA runtime setup failed: %s", e)

WEB_DIRECTORY = "./web"

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
