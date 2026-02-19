"""
Audio source separation module using Demucs.
Wraps the demucs Python API with caching and error recovery.
"""

from pathlib import Path

try:
    from demucs.api import Separator
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

from melody_extractor.utils import compute_file_hash, save_audio

DEMUCS_MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]


def get_available_models() -> list[str]:
    """Return the list of available Demucs models if Demucs is installed."""
    if DEMUCS_AVAILABLE:
        return DEMUCS_MODELS
    return []


def separate_audio(
    audio_path: str,
    model_name: str = "htdemucs",
    device: str = "cpu",
    cache_dir: str = "./cache",
) -> dict:
    """
    Separate audio into instrumental and vocals stems using Demucs.

    Args:
        audio_path: Path to the input audio file.
        model_name: Demucs model to use for separation.
        device: Compute device ("cpu" or "cuda").
        cache_dir: Directory to store cached separation results.

    Returns:
        A dict with keys:
            - "instrumental_path": str path to the instrumental WAV (or original on error)
            - "vocals_path": str path to the vocals WAV (or None on error)
            - "cached": bool indicating whether the result came from cache
            - "error": str error message (only present on failure)
    """
    if not DEMUCS_AVAILABLE:
        return {
            "instrumental_path": audio_path,
            "vocals_path": None,
            "error": "Demucs not installed",
            "cached": False,
        }

    audio_bytes = Path(audio_path).read_bytes()
    file_hash = compute_file_hash(audio_bytes)

    cache_base = Path(cache_dir) / file_hash / model_name
    instrumental_cache = cache_base / "instrumental.wav"
    vocals_cache = cache_base / "vocals.wav"

    if instrumental_cache.exists() and vocals_cache.exists():
        return {
            "instrumental_path": str(instrumental_cache),
            "vocals_path": str(vocals_cache),
            "cached": True,
        }

    def _run_separation(sep_device: str) -> dict:
        separator = Separator(model=model_name, device=sep_device, progress=True)
        _origin, separated = separator.separate_audio_file(audio_path)

        instrumental_tensor = (
            separated["drums"] + separated["bass"] + separated["other"]
        )
        vocals_tensor = separated["vocals"]

        # Shape: (channels, samples) → transpose to (samples, channels)
        instrumental_np = instrumental_tensor.numpy().T
        vocals_np = vocals_tensor.numpy().T

        cache_base.mkdir(parents=True, exist_ok=True)

        save_audio(instrumental_np, 44100, str(instrumental_cache))
        save_audio(vocals_np, 44100, str(vocals_cache))

        return {
            "instrumental_path": str(instrumental_cache),
            "vocals_path": str(vocals_cache),
            "cached": False,
        }

    try:
        return _run_separation(device)
    except RuntimeError as e:
        err_str = str(e)
        if "CUDA" in err_str or "out of memory" in err_str:
            try:
                return _run_separation("cpu")
            except Exception as fallback_exc:
                return {
                    "instrumental_path": audio_path,
                    "vocals_path": None,
                    "error": str(fallback_exc),
                    "cached": False,
                }
        return {
            "instrumental_path": audio_path,
            "vocals_path": None,
            "error": err_str,
            "cached": False,
        }
    except Exception as e:
        return {
            "instrumental_path": audio_path,
            "vocals_path": None,
            "error": str(e),
            "cached": False,
        }
