import os
import platform
from typing import Any

if platform.system() == "Darwin":
    _ = os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch as _torch
except Exception:
    _torch = None


def _mps_backend() -> Any | None:
    if _torch is None:
        return None
    return getattr(_torch.backends, "mps", None)


def _mps_available() -> bool:
    mps_backend = _mps_backend()
    return bool(mps_backend is not None and mps_backend.is_available())


def select_torch_device(requested: str = "auto") -> str:
    normalized = requested.lower()

    if _torch is None:
        return "cpu"

    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda" if _torch.cuda.is_available() else "cpu"
    if normalized == "mps":
        return "mps" if _mps_available() else "cpu"

    if platform.system() == "Darwin" and _mps_available():
        return "mps"
    if _torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def get_torch_runtime_info() -> dict[str, bool | str]:
    if _torch is None:
        return {
            "torch_available": False,
            "cuda_available": False,
            "mps_built": False,
            "mps_available": False,
            "selected_device": "cpu",
        }

    mps_backend = _mps_backend()
    mps_built = bool(mps_backend is not None and mps_backend.is_built())
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    return {
        "torch_available": True,
        "cuda_available": bool(_torch.cuda.is_available()),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "selected_device": select_torch_device("auto"),
    }
