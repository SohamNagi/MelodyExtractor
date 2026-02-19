"""
Utility functions for the MelodyExtractor pipeline.

Provides caching, audio I/O, frequency/MIDI conversions, and parameter hashing.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile
import pretty_midi


def compute_file_hash(data: bytes) -> str:
    """
    Compute SHA256 hash of input bytes.

    Args:
        data: Raw bytes to hash

    Returns:
        Hexadecimal SHA256 digest
    """
    return hashlib.sha256(data).hexdigest()


def get_cache_dir(base: str = "./cache") -> Path:
    """
    Get or create cache directory.

    Args:
        base: Base cache directory path (default: "./cache")

    Returns:
        Path object for cache directory (created if it doesn't exist)
    """
    cache_dir = Path(base)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(file_hash: str, step: str, params_hash: str = "") -> Path:
    """
    Get cache path for a specific processing step.

    Args:
        file_hash: Hash of the input file
        step: Processing step name (e.g., 'spectrogram', 'pitch')
        params_hash: Hash of processing parameters (optional)

    Returns:
        Path object for cache location (creates directories if needed)
    """
    if params_hash:
        cache_path = Path("cache") / file_hash / f"{step}_{params_hash}"
    else:
        cache_path = Path("cache") / file_hash / step

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_audio(source: str | Path | bytes, sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    Load audio from file path or raw bytes.

    Args:
        source: File path (str or Path) or raw audio bytes
        sr: Target sample rate (default: 44100)

    Returns:
        Tuple of (audio_data as float32, sample rate)

    Raises:
        ValueError: If source is empty bytes or file cannot be loaded
    """
    if isinstance(source, bytes):
        if len(source) == 0:
            raise ValueError("Audio bytes cannot be empty")

        # Write bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        try:
            audio, loaded_sr = librosa.load(tmp_path, sr=sr, mono=True)
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink()
    else:
        # Load from file path
        audio, loaded_sr = librosa.load(str(source), sr=sr, mono=True)

    return audio.astype(np.float32), loaded_sr


def save_audio(audio: np.ndarray, sr: int, path: str | Path) -> None:
    """
    Save audio array to file.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        path: Output file path
    """
    soundfile.write(str(path), audio, sr)


def hz_to_midi(freq: float | np.ndarray) -> float | np.ndarray:
    """
    Convert frequency in Hz to MIDI note number.

    Vectorized to handle arrays. Handles edge cases:
    - 0 Hz → NaN
    - NaN → NaN
    - Negative frequencies → NaN

    Formula: MIDI = 69 + 12 * log2(freq / 440)

    Args:
        freq: Frequency or array of frequencies in Hz

    Returns:
        MIDI note number(s) as float or array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 69.0 + 12.0 * np.log2(freq / 440.0)

    return result


def midi_to_note_name(midi_num: int) -> str:
    """
    Convert MIDI note number to note name.

    Args:
        midi_num: MIDI note number (0-127)

    Returns:
        Note name (e.g., 'C4', 'A#5')
    """
    return pretty_midi.note_number_to_name(midi_num)


def format_time(seconds: float) -> str:
    """
    Format time duration as "mm:ss.sss".

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string (e.g., "01:23.456")
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def hash_params(**kwargs) -> str:
    """
    Create a hash of processing parameters.

    Args:
        **kwargs: Parameter key-value pairs

    Returns:
        First 12 characters of SHA256 hex digest
    """
    # Sort kwargs for deterministic ordering
    sorted_items = sorted(kwargs.items())

    # JSON serialize with default handler for non-serializable types
    json_str = json.dumps(sorted_items, default=str, separators=(',', ':'))

    # Compute SHA256 and return first 12 chars
    hash_result = hashlib.sha256(json_str.encode()).hexdigest()
    return hash_result[:12]


def estimate_tempo_bpm(audio: np.ndarray, sr: int) -> float:
    """
    Estimate global tempo (BPM) from an audio signal.

    Args:
        audio: Mono audio signal
        sr: Sample rate

    Returns:
        Estimated tempo in BPM. Falls back to 120.0 when estimation is invalid.
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    if isinstance(tempo, np.ndarray):
        if tempo.size == 0:
            return 120.0
        tempo_value = float(tempo.item(0))
    else:
        tempo_value = float(tempo)

    if not np.isfinite(tempo_value) or tempo_value <= 0.0:
        return 120.0

    return tempo_value
