"""
Key detection module for estimating musical key and scale from audio.
Implements both Essentia (primary) and librosa (fallback) approaches.
"""

import numpy as np
import librosa

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

# Krumhansl-Schmuckler key profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_key_essentia(audio: np.ndarray, sr: int = 44100) -> dict:
    """Detect musical key using Essentia's KeyExtractor algorithm.

    Args:
        audio: Mono float32 audio array.
        sr: Sample rate of the audio.

    Returns:
        Dictionary with key, scale, confidence, and method.
    """
    key_extractor = es.KeyExtractor(sampleRate=float(sr))
    key, scale, strength = key_extractor(audio.astype(np.float32))
    return {
        "key": key,
        "scale": scale,
        "confidence": float(strength),
        "method": "essentia",
    }


def detect_key_librosa(audio: np.ndarray, sr: int = 44100) -> dict:
    """Detect musical key using the Krumhansl-Schmuckler algorithm via librosa.

    Computes a chroma vector and correlates it against all 12 rotations of the
    major and minor key profiles to find the best matching key.

    Args:
        audio: Mono float32 audio array.
        sr: Sample rate of the audio.

    Returns:
        Dictionary with key, scale, confidence, and method.
    """
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)

    best_corr = -np.inf
    best_idx = 0
    best_scale = "major"

    for i in range(12):
        major_profile = np.roll(KS_MAJOR, -i)
        minor_profile = np.roll(KS_MINOR, -i)

        corr_major = np.corrcoef(chroma_mean, major_profile)[0, 1]
        corr_minor = np.corrcoef(chroma_mean, minor_profile)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_idx = i
            best_scale = "major"

        if corr_minor > best_corr:
            best_corr = corr_minor
            best_idx = i
            best_scale = "minor"

    confidence = float(np.clip(np.abs(best_corr), 0.0, 1.0))

    return {
        "key": NOTE_NAMES[best_idx],
        "scale": best_scale,
        "confidence": confidence,
        "method": "librosa",
    }


def detect_key(audio: np.ndarray, sr: int = 44100, method: str = "auto") -> dict:
    """Detect musical key and scale from audio.

    Args:
        audio: Mono float32 audio array.
        sr: Sample rate of the audio.
        method: Detection method — "auto", "essentia", or "librosa".
                "auto" tries essentia first, falls back to librosa.

    Returns:
        Dictionary with key, scale, confidence, method, and optional error.
    """
    try:
        if method == "auto":
            if ESSENTIA_AVAILABLE:
                return detect_key_essentia(audio, sr)
            return detect_key_librosa(audio, sr)
        elif method == "essentia":
            if not ESSENTIA_AVAILABLE:
                raise ImportError(
                    "Essentia is not installed. Install it or use method='librosa'."
                )
            return detect_key_essentia(audio, sr)
        elif method == "librosa":
            return detect_key_librosa(audio, sr)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from: auto, essentia, librosa.")
    except Exception as e:
        return {
            "key": "Unknown",
            "scale": "",
            "confidence": 0.0,
            "method": "error",
            "error": str(e),
        }


def get_available_methods() -> list[str]:
    """Return a sorted list of available key detection methods.

    Returns:
        List containing at minimum "auto" and "librosa", plus "essentia" if available.
    """
    methods = ["auto", "librosa"]
    if ESSENTIA_AVAILABLE:
        methods.append("essentia")
    return sorted(methods)
