"""CREPE neural pitch tracker via torchcrepe."""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from melody_extractor.extractors.base import MelodyExtractor
from melody_extractor.torch_backend import select_torch_device

torch: Any | None = None
torchcrepe: Any | None = None
_crepe_available = False

try:
    import torch  # type: ignore[import]
    import torchcrepe  # type: ignore[import]

    _crepe_available = True
except Exception:
    pass

_CREPE_AVAILABLE = _crepe_available


class CrepeExtractor(MelodyExtractor):
    name: str = "CREPE (torchcrepe)"
    description: str = "Neural pitch tracker, strong for monophonic"
    available: bool = _CREPE_AVAILABLE

    def extract(
        self, audio: NDArray[np.float64], sr: int, **kwargs
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Extract melody using CREPE via torchcrepe.

        Parameters
        ----------
        audio : np.ndarray
            Mono audio as a 1D float array.
        sr : int
            Sample rate in Hz.
        **kwargs
            hop_length : int, default 256
            fmin : float, default 98.0
            fmax : float, default 1400.0
            model : str, default "tiny"
            confidence_threshold : float, default 0.18
            batch_size : int, default 256

        Returns
        -------
        times : np.ndarray, shape (T,)
        f0_hz : np.ndarray, shape (T,) — NaN for unvoiced frames
        confidence : np.ndarray, shape (T,) — periodicity in [0, 1]
        """
        hop_length = int(kwargs.get("hop_length", 256))
        fmin = float(kwargs.get("fmin", 98.0))
        fmax = float(kwargs.get("fmax", 1400.0))
        model = str(kwargs.get("model", "tiny"))
        confidence_threshold = float(kwargs.get("confidence_threshold", 0.18))
        batch_size = int(kwargs.get("batch_size", 256))

        if torch is None or torchcrepe is None:
            raise RuntimeError("torchcrepe is not available")
        torch_mod = cast(Any, torch)
        torchcrepe_mod = cast(Any, torchcrepe)

        selected_device = select_torch_device("auto")

        # torchcrepe expects shape (1, N)
        audio_tensor_cpu = torch_mod.tensor(audio)[None].float().to("cpu")

        def _predict_with_device(device_name: str) -> tuple[Any, Any]:
            return torchcrepe_mod.predict(
                audio_tensor_cpu.to(device_name),
                sr,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                model=model,
                batch_size=batch_size,
                device=device_name,
                return_periodicity=True,
            )

        try:
            pitch, periodicity = _predict_with_device(selected_device)
        except Exception as primary_exc:
            if selected_device != "cpu":
                pitch, periodicity = _predict_with_device("cpu")
            else:
                raise RuntimeError(
                    f"torchcrepe inference failed on {selected_device}: {primary_exc}") from primary_exc

        # Smooth periodicity with a median filter
        periodicity = torchcrepe_mod.filter.median(periodicity, 3)

        # Threshold: frames below confidence_threshold become NaN in pitch
        pitch, periodicity = torchcrepe_mod.threshold.At(confidence_threshold)(
            pitch, periodicity
        )

        f0 = pitch.squeeze(0).cpu().numpy().astype(np.float64)
        confidence = periodicity.squeeze(0).cpu().numpy().astype(np.float64)

        times = np.arange(len(f0), dtype=np.float64) * hop_length / sr

        return times, f0, confidence

    def get_default_params(self) -> dict[str, Any]:
        return {
            "hop_length": 256,
            "fmin": 98.0,
            "fmax": 1400.0,
            "model": "tiny",
            "confidence_threshold": 0.18,
            "batch_size": 256,
        }

    def get_param_descriptions(self) -> dict[str, str]:
        return {
            "hop_length": "Analysis hop length in samples",
            "fmin": "Minimum frequency in Hz",
            "fmax": "Maximum frequency in Hz",
            "model": "Model capacity: 'tiny' or 'full'",
            "confidence_threshold": "Periodicity threshold below which frames are unvoiced",
            "batch_size": "Batch size for GPU inference",
        }
