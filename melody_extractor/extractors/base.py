"""Abstract base class for melody extraction algorithms."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MelodyExtractor(ABC):
    name: str = ""
    description: str = ""
    available: bool = True

    @abstractmethod
    def extract(
        self, audio: np.ndarray, sr: int, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract melody from audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio signal as a 1D float32/float64 array.
        sr : int
            Sample rate in Hz.
        **kwargs
            Algorithm-specific parameters (see get_default_params).

        Returns
        -------
        times : np.ndarray, shape (T,)
            Frame timestamps in seconds.
        f0_hz : np.ndarray, shape (T,)
            Fundamental frequency in Hz. NaN for unvoiced frames.
        confidence : np.ndarray, shape (T,)
            Per-frame confidence in [0, 1]. Ones if unavailable.
        """
        ...

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """Return default algorithm parameters as a flat dict."""
        ...

    def get_param_descriptions(self) -> dict[str, str]:
        """Return human-readable descriptions for each parameter key."""
        return {}
