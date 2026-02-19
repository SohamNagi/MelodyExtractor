"""
Post-processing module: converts frame-level pitch data into discrete note events.

Pipeline: confidence threshold → smooth pitch → hz_to_midi → quantize → segment → velocity
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def apply_confidence_threshold(
    f0: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Zero out (NaN) f0 frames where confidence falls below threshold.

    Args:
        f0: Frame-level pitch in Hz, shape (T,). NaN indicates unvoiced.
        confidence: Frame-level confidence in [0, 1], shape (T,).
        threshold: Frames with confidence < threshold are set to NaN.

    Returns:
        Filtered f0 array (copy); original is not mutated.
    """
    result = f0.copy().astype(float)
    result[confidence < threshold] = np.nan
    return result


def smooth_pitch(f0: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply median filter to voiced (non-NaN) regions of the pitch array.

    NaN positions are restored after filtering so unvoiced gaps are preserved.

    Args:
        f0: Frame-level pitch in Hz, shape (T,). NaN indicates unvoiced.
        window: Median filter kernel size (frames). Skipped if <= 1.

    Returns:
        Smoothed f0 array (copy).
    """
    result = f0.copy().astype(float)

    if window <= 1:
        return result

    nan_mask = np.isnan(result)

    # Replace NaN with 0 for filtering, then restore NaN positions
    temp = result.copy()
    temp[nan_mask] = 0.0
    filtered = median_filter(temp, size=window)
    result = filtered.astype(float)
    result[nan_mask] = np.nan

    return result


def hz_to_midi_array(f0: np.ndarray) -> np.ndarray:
    """Convert Hz pitch array to MIDI note numbers (floats).

    Frames where f0 is NaN, zero, or negative map to NaN in the output.

    Args:
        f0: Frame-level pitch in Hz, shape (T,).

    Returns:
        MIDI float array, shape (T,). NaN for unvoiced/invalid frames.
    """
    f0 = np.array(f0, dtype=float)
    midi = np.full_like(f0, np.nan)

    valid = np.isfinite(f0) & (f0 > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        midi[valid] = 69.0 + 12.0 * np.log2(f0[valid] / 440.0)

    return midi


def quantize_pitch(midi_notes: np.ndarray, mode: str = "semitone") -> np.ndarray:
    """Quantize MIDI float array to desired pitch resolution.

    Args:
        midi_notes: MIDI float array, shape (T,). NaN values are preserved.
        mode: One of:
            - "semitone": round to nearest integer semitone.
            - "quarter":  round to nearest quarter-tone (0.5 semitone).
            - "none":     return unchanged copy.

    Returns:
        Quantized MIDI array (copy).
    """
    result = midi_notes.copy().astype(float)

    if mode == "semitone":
        valid = ~np.isnan(result)
        result[valid] = np.round(result[valid])
    elif mode == "quarter":
        valid = ~np.isnan(result)
        result[valid] = np.round(result[valid] * 2) / 2
    elif mode == "none":
        pass  # already copied
    else:
        raise ValueError(f"Unknown quantize mode: {mode!r}. Choose 'semitone', 'quarter', or 'none'.")

    return result


def segment_notes(
    times: np.ndarray,
    midi_notes: np.ndarray,
    confidence: np.ndarray,
    min_note_length: float = 0.05,
) -> list[dict]:
    """Group consecutive frames of the same pitch into discrete note events.

    NaN frames act as gaps: they terminate the current note and are skipped.

    Args:
        times: Frame timestamps in seconds, shape (T,).
        midi_notes: Quantized MIDI float array, shape (T,). NaN = unvoiced.
        confidence: Frame-level confidence in [0, 1], shape (T,).
        min_note_length: Notes shorter than this duration (seconds) are discarded.

    Returns:
        List of note dicts::

            {
                "start": float,          # seconds
                "end": float,            # seconds
                "pitch": int,            # MIDI note [0, 127]
                "velocity": 80,          # placeholder; updated by assign_velocity
                "confidence_avg": float, # mean confidence of contributing frames
            }
    """
    frame_dur = (times[1] - times[0]) if len(times) > 1 else 0.01

    notes: list[dict] = []
    group_start_idx: int | None = None
    group_pitch_int: int | None = None
    group_midi_values: list[float] = []
    group_confidences: list[float] = []

    def _flush_group(last_idx: int) -> None:
        if group_start_idx is None:
            return
        start_time = float(times[group_start_idx])
        end_time = float(times[last_idx]) + frame_dur
        duration = end_time - start_time
        if duration < min_note_length:
            return
        pitch = int(round(float(np.median(group_midi_values))))
        pitch = max(0, min(127, pitch))
        conf_avg = float(np.mean(group_confidences))
        notes.append(
            {
                "start": start_time,
                "end": end_time,
                "pitch": pitch,
                "velocity": 80,
                "confidence_avg": conf_avg,
            }
        )

    for i, (t, m, c) in enumerate(zip(times, midi_notes, confidence)):
        if np.isnan(m):
            # Gap: flush current group
            _flush_group(i - 1)
            group_start_idx = None
            group_pitch_int = None
            group_midi_values = []
            group_confidences = []
            continue

        current_pitch_int = int(round(float(m)))

        if group_pitch_int is None:
            # Start new group
            group_start_idx = i
            group_pitch_int = current_pitch_int
            group_midi_values = [float(m)]
            group_confidences = [float(c)]
        elif current_pitch_int == group_pitch_int:
            # Continue current group
            group_midi_values.append(float(m))
            group_confidences.append(float(c))
        else:
            # Pitch changed: flush and start new group
            _flush_group(i - 1)
            group_start_idx = i
            group_pitch_int = current_pitch_int
            group_midi_values = [float(m)]
            group_confidences = [float(c)]

    # Flush any remaining group at end of array
    if group_start_idx is not None:
        _flush_group(len(times) - 1)

    return notes


def assign_velocity(notes: list[dict], method: str = "from_confidence") -> list[dict]:
    """Set the velocity field of each note dict.

    Args:
        notes: List of note dicts (as produced by :func:`segment_notes`).
        method: One of:
            - "from_confidence": velocity = 40 + confidence_avg * 80, clamped to [1, 127].
            - "fixed": velocity stays at its current value (default 80).

    Returns:
        The same list with velocity fields updated (modified in place).
    """
    if method == "from_confidence":
        for note in notes:
            raw = int(40 + note["confidence_avg"] * 80)
            note["velocity"] = max(1, min(127, raw))
    elif method == "fixed":
        pass  # keep existing velocity value
    else:
        raise ValueError(f"Unknown velocity method: {method!r}. Choose 'from_confidence' or 'fixed'.")

    return notes


def postprocess_pipeline(
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray | None,
    confidence_threshold: float = 0.5,
    smoothing_window: int = 5,
    quantize: str = "semitone",
    min_note_length: float = 0.05,
    velocity_method: str = "from_confidence",
) -> list[dict]:
    """Full post-processing pipeline from raw frame data to note events.

    Steps (in order):
        1. Apply confidence threshold  (unconfident frames → NaN)
        2. Smooth pitch                (median filter voiced regions)
        3. Convert Hz → MIDI floats
        4. Quantize pitch
        5. Segment into note events
        6. Assign velocities

    Args:
        times: Frame timestamps in seconds, shape (T,).
        f0: Frame-level pitch in Hz, shape (T,). NaN = unvoiced.
        confidence: Frame-level confidence in [0, 1], shape (T,), or None
            (treated as all-ones).
        confidence_threshold: Frames below this confidence are silenced.
        smoothing_window: Median filter size in frames (<=1 = no smoothing).
        quantize: Pitch quantization mode passed to :func:`quantize_pitch`.
        min_note_length: Minimum note duration in seconds.
        velocity_method: Velocity assignment method passed to :func:`assign_velocity`.

    Returns:
        List of note dicts, or empty list if no notes were detected.
    """
    if confidence is None:
        confidence = np.ones_like(f0, dtype=float)

    # Step 1: confidence threshold
    f0_filtered = apply_confidence_threshold(f0, confidence, threshold=confidence_threshold)

    # Step 2: smooth pitch
    f0_smooth = smooth_pitch(f0_filtered, window=smoothing_window)

    # Step 3: Hz → MIDI
    midi = hz_to_midi_array(f0_smooth)

    # Step 4: quantize
    midi_q = quantize_pitch(midi, mode=quantize)

    # Step 5: segment
    notes = segment_notes(times, midi_q, confidence, min_note_length=min_note_length)

    if not notes:
        return []

    # Step 6: velocity
    notes = assign_velocity(notes, method=velocity_method)

    return notes
