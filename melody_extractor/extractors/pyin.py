"""librosa pYIN probabilistic pitch tracker."""

from typing import Any

import librosa
import numpy as np
from numpy.typing import NDArray

from melody_extractor.extractors.base import MelodyExtractor


class PYINExtractor(MelodyExtractor):
    name: str = "librosa pYIN"
    description: str = "Baseline probabilistic pitch tracker"
    available: bool = True

    def extract(
        self,
        audio: NDArray[np.float64],
        sr: int,
        **kwargs: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Extract melody using librosa's pYIN algorithm.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio as a 1D float array.
        sr : int
            Sample rate in Hz.
        **kwargs
            hop_length : int, default 512
            fmin : float, default 98.0  (G2)
            fmax : float, default 1400.0 (~F6)

        Returns
        -------
        times : np.ndarray, shape (T,)
        f0_hz : np.ndarray, shape (T,) — NaN for unvoiced frames (native pYIN behaviour)
        confidence : np.ndarray, shape (T,) — voiced probabilities in [0, 1]
        """
        hop_length = int(kwargs.get("hop_length", 512))
        fmin = float(kwargs.get("fmin", 98.0))
        fmax = float(kwargs.get("fmax", 1400.0))

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        if voiced_probs is not None:
            confidence = voiced_probs.astype(np.float64)
        else:
            confidence = np.ones(len(f0), dtype=np.float64)

        f0 = f0.astype(np.float64)

        return times, f0, confidence

    def get_default_params(self) -> dict[str, Any]:
        return {
            "hop_length": 512,
            "fmin": 98.0,
            "fmax": 1400.0,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        return {
            "hop_length": "Analysis hop length in samples",
            "fmin": "Minimum frequency in Hz (default G2 = 98 Hz)",
            "fmax": "Maximum frequency in Hz (default ~F6 = 1400 Hz)",
        }
