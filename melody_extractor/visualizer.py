import streamlit as st
import numpy as np

from melody_extractor.midi_gen import midi_bytes_to_wav_bytes
from melody_extractor.utils import load_audio, save_audio


def render_midi_player(
    midi_bytes: bytes,
    height: int = 500,
    target_duration_sec: float | None = None,
    sample_rate: int = 44100,
) -> None:
    """
    Render synthesized WAV playback from MIDI bytes.

    Args:
        midi_bytes: Raw MIDI file bytes
        height: Kept for API compatibility (unused)
        target_duration_sec: Optional target duration in seconds to match original audio length
        sample_rate: Sample rate for MIDI synthesis (default 44100)
    """
    _ = height
    wav_bytes = midi_bytes_to_wav_bytes(midi_bytes, sample_rate=sample_rate)

    if isinstance(target_duration_sec, (int, float)) and target_duration_sec > 0.0:
        audio, sr = load_audio(wav_bytes, sr=sample_rate)
        target_samples = int(round(float(target_duration_sec) * sr))

        if target_samples > 0:
            if len(audio) < target_samples:
                pad_width = target_samples - len(audio)
                audio = np.pad(audio, (0, pad_width), mode="constant")
            elif len(audio) > target_samples:
                audio = audio[:target_samples]

            temp_path = st.session_state.get("_midi_preview_wav_path")
            if not isinstance(temp_path, str) or len(temp_path) == 0:
                import tempfile

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_path = tmp.name
                tmp.close()
                st.session_state["_midi_preview_wav_path"] = temp_path

            save_audio(audio, sr, temp_path)
            st.audio(temp_path, format="audio/wav")
            return

    st.audio(wav_bytes, format="audio/wav")


def render_audio_player(audio_bytes: bytes, label: str = "Audio") -> None:
    """
    Render a simple audio player for WAV files.

    Args:
        audio_bytes: Raw audio file bytes (WAV format)
        label: Label for the audio player (unused but kept for API consistency)
    """
    st.audio(audio_bytes, format="audio/wav")
