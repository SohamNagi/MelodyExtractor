"""
Audio source separation module using Demucs.
Wraps the demucs Python API (v4.0.x) with caching and error recovery.
"""

from pathlib import Path

torch = None
get_model = None
apply_model = None
AudioFile = None
_demucs_available = False

try:
    import torch
    from demucs.pretrained import get_model as _get_model
    from demucs.apply import apply_model as _apply_model
    from demucs.audio import AudioFile as _AudioFile

    get_model = _get_model
    apply_model = _apply_model
    AudioFile = _AudioFile
    _demucs_available = True
except ImportError:
    pass

DEMUCS_AVAILABLE = _demucs_available

from melody_extractor.utils import compute_file_hash, save_audio
from melody_extractor.torch_backend import select_torch_device

DEMUCS_MODELS = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]


def get_available_models() -> list[str]:
    """Return the list of available Demucs models if Demucs is installed."""
    if DEMUCS_AVAILABLE:
        return DEMUCS_MODELS
    return []


def separate_audio(
    audio_path: str,
    model_name: str = "htdemucs",
    device: str = "auto",
    cache_dir: str = "./cache",
) -> dict[str, str | bool | None]:
    if not DEMUCS_AVAILABLE:
        return {
            "instrumental_path": audio_path,
            "vocals_path": None,
            "error": "Demucs not installed",
            "cached": False,
        }

    audio_bytes = Path(audio_path).read_bytes()
    file_hash = compute_file_hash(audio_bytes)
    resolved_device = select_torch_device(device)

    cache_base = Path(cache_dir) / file_hash / model_name
    instrumental_cache = cache_base / "instrumental.wav"
    vocals_cache = cache_base / "vocals.wav"

    if instrumental_cache.exists() and vocals_cache.exists():
        return {
            "instrumental_path": str(instrumental_cache),
            "vocals_path": str(vocals_cache),
            "cached": True,
        }

    def _run_separation(sep_device: str) -> dict[str, str | bool | None]:
        if get_model is None or apply_model is None or AudioFile is None:
            raise RuntimeError("Demucs runtime is unavailable")
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
        return _run_separation(resolved_device)
    except Exception as e:
        if resolved_device != "cpu":
            try:
                return _run_separation("cpu")
            except Exception as fallback_exc:
                return {
                    "instrumental_path": audio_path,
                    "vocals_path": None,
                    "error": f"{e} | CPU fallback failed: {fallback_exc}",
                    "cached": False,
                }
        return {
            "instrumental_path": audio_path,
            "vocals_path": None,
            "error": str(e),
            "cached": False,
        }
