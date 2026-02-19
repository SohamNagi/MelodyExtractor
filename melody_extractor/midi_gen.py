import io
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi


def notes_to_midi(
    notes: list[dict], tempo: float = 120.0, program: int = 0
) -> pretty_midi.PrettyMIDI:
    """
    Convert note events to a PrettyMIDI object.

    Args:
        notes: List of note dictionaries with keys: start, end, pitch, velocity, confidence_avg
        tempo: Tempo in BPM (default 120.0)
        program: MIDI program number (default 0 = Acoustic Grand Piano)

    Returns:
        PrettyMIDI object with notes added to an instrument
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=960)
    inst = pretty_midi.Instrument(program=program, name="Melody")

    sorted_notes = sorted(notes, key=lambda item: (float(item["start"]), float(item["end"])))

    for note in sorted_notes:
        inst.notes.append(
            pretty_midi.Note(
                velocity=note["velocity"],
                pitch=int(note["pitch"]),
                start=float(note["start"]),
                end=float(note["end"]),
            )
        )

    pm.instruments.append(inst)
    return pm


def midi_to_bytes(pm: pretty_midi.PrettyMIDI) -> bytes:
    """
    Convert a PrettyMIDI object to bytes.

    Args:
        pm: PrettyMIDI object

    Returns:
        MIDI data as bytes
    """
    bio = io.BytesIO()
    pm.write(bio)
    bio.seek(0)
    return bio.read()


def save_midi(pm: pretty_midi.PrettyMIDI, path: str | Path) -> None:
    """
    Save a PrettyMIDI object to a file.

    Args:
        pm: PrettyMIDI object
        path: File path to save MIDI to
    """
    pm.write(str(path))


def notes_to_dataframe(notes: list[dict]) -> pd.DataFrame:
    """
    Convert note events to a pandas DataFrame.

    Args:
        notes: List of note dictionaries with keys: start, end, pitch, velocity, confidence_avg

    Returns:
        DataFrame with columns: start, end, duration, pitch, note_name, velocity, confidence
    """
    if not notes:
        return pd.DataFrame(
            columns=["start", "end", "duration", "pitch", "note_name", "velocity", "confidence"]
        )

    data = []
    for note in notes:
        duration = note["end"] - note["start"]
        note_name = pretty_midi.note_number_to_name(int(note["pitch"]))
        data.append(
            {
                "start": note["start"],
                "end": note["end"],
                "duration": duration,
                "pitch": note["pitch"],
                "note_name": note_name,
                "velocity": note["velocity"],
                "confidence": note["confidence_avg"],
            }
        )

    return pd.DataFrame(data)


def f0_to_dataframe(
    times: np.ndarray, f0: np.ndarray, confidence: np.ndarray
) -> pd.DataFrame:
    """
    Convert F0 contour to a pandas DataFrame.

    Args:
        times: Time values in seconds
        f0: Fundamental frequency values in Hz
        confidence: Confidence values for each F0 estimate

    Returns:
        DataFrame with columns: time, f0_hz, confidence
    """
    return pd.DataFrame({
        "time": times,
        "f0_hz": f0,
        "confidence": confidence,
    })
