"""Essentia PredominantPitchMelodia melody extractor."""

from typing import Any

import numpy as np

from melody_extractor.extractors.base import MelodyExtractor

try:
    import essentia.standard as es  # type: ignore[import]

    _ESSENTIA_AVAILABLE = True
except Exception:
    _ESSENTIA_AVAILABLE = False


class MelodiaExtractor(MelodyExtractor):
    name: str = "Essentia PredominantPitchMelodia"
    description: str = "Best for predominant melody in complex mixes"
    available: bool = _ESSENTIA_AVAILABLE

    def extract(
        self, audio: np.ndarray, sr: int, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract melody using Essentia's PredominantPitchMelodia.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio as a 1D float32 array (Essentia expects float32).
        sr : int
            Sample rate in Hz.
        **kwargs
            hop_size : int, default 128
            frame_size : int, default 2048
            fmin : float, default 80.0
            fmax : float, default 20000.0

        Returns
        -------
        times : np.ndarray, shape (T,)
        f0_hz : np.ndarray, shape (T,) — NaN for unvoiced
        confidence : np.ndarray, shape (T,) — ones (Melodia provides no per-frame confidence)
        """
        hop_size = int(kwargs.get("hop_size", 128))
        frame_size = int(kwargs.get("frame_size", 2048))
        fmin = float(kwargs.get("fmin", 80.0))
        fmax = float(kwargs.get("fmax", 20000.0))

        audio_f32 = audio.astype(np.float32)

        equal_loudness = es.EqualLoudness(sampleRate=sr)
        audio_eq = equal_loudness(audio_f32)

        pitch_extractor = es.PredominantPitchMelodia(
            hopSize=hop_size,
            frameSize=frame_size,
            minFrequency=fmin,
            maxFrequency=fmax,
            sampleRate=sr,
        )
        pitch, pitch_confidence = pitch_extractor(audio_eq)

        pitch = np.array(pitch, dtype=np.float64)

        # Essentia uses 0.0 to mark unvoiced frames — convert to NaN
        pitch[pitch == 0.0] = np.nan

        times = np.arange(len(pitch), dtype=np.float64) * hop_size / sr
        confidence = np.ones(len(pitch), dtype=np.float64)

        return times, pitch, confidence

    def get_default_params(self) -> dict[str, Any]:
        return {
            "hop_size": 128,
            "frame_size": 2048,
            "fmin": 80.0,
            "fmax": 20000.0,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        return {
            "hop_size": "Analysis hop size in samples",
            "frame_size": "Analysis frame size in samples",
            "fmin": "Minimum frequency in Hz",
            "fmax": "Maximum frequency in Hz",
        }
