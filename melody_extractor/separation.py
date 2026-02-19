"""
Audio source separation module using Demucs.
Wraps the demucs Python API (v4.0.x) with caching and error recovery.
"""

from pathlib import Path

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
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
        model = get_model(model_name)
        model.to(sep_device)

        audio_file = AudioFile(Path(audio_path))
        sr = model.samplerate
        wav = audio_file.read(samplerate=sr, channels=model.audio_channels)
        wav = wav.to(sep_device)

        # apply_model expects (batch, channels, samples)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)

        sources = apply_model(model, wav, device=sep_device, progress=True)
        # sources shape: (batch, num_sources, channels, samples)
        sources = sources.squeeze(0)  # (num_sources, channels, samples)

        # Map source names to tensors
        source_names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        source_dict = dict(zip(source_names, sources))

        # Build instrumental = everything except vocals
        instrumental_parts = [t for name, t in source_dict.items() if name != "vocals"]
        instrumental_tensor = instrumental_parts[0]
        for part in instrumental_parts[1:]:
            instrumental_tensor = instrumental_tensor + part
        vocals_tensor = source_dict["vocals"]

        # Shape: (channels, samples) → transpose to (samples, channels)
        instrumental_np = instrumental_tensor.cpu().numpy().T
        vocals_np = vocals_tensor.cpu().numpy().T

        cache_base.mkdir(parents=True, exist_ok=True)

        save_audio(instrumental_np, sr, str(instrumental_cache))
        save_audio(vocals_np, sr, str(vocals_cache))

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
