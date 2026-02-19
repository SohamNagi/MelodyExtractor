import base64
import streamlit as st
import streamlit.components.v1

MIDI_PLAYER_CDN = (
    "https://cdn.jsdelivr.net/combine/"
    "npm/tone@14.7.58,"
    "npm/@magenta/music@1.23.1/es6/core.js,"
    "npm/focus-visible@5,"
    "npm/html-midi-player@1.5.0"
)


def render_midi_player(midi_bytes: bytes, height: int = 500) -> None:
    """
    Render an embedded MIDI player with synced piano-roll visualizer.

    Args:
        midi_bytes: Raw MIDI file bytes
        height: Total height of the player + visualizer in pixels (default 500)
    """
    data_url = "data:audio/midi;base64," + base64.b64encode(midi_bytes).decode("utf-8")

    html_string = f"""
    <script src="{MIDI_PLAYER_CDN}"></script>
    <style>
        midi-player {{
            display: block;
            width: 100%;
            background: #0e1117;
            color: #e6edf3;
            border-radius: 8px;
            border: 1px solid #30363d;
            margin-bottom: 12px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}

        midi-player::part(control-panel) {{
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border-bottom: 1px solid #21262d;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        midi-player::part(play-button) {{
            background: #ff4b4b;
            border: none;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }}

        midi-player::part(play-button):hover {{
            background: #ff3838;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
        }}

        midi-player::part(play-button):active {{
            transform: scale(0.95);
        }}

        midi-player::part(time-display) {{
            color: #79c0ff;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            min-width: 100px;
            text-align: center;
        }}

        midi-player::part(slider) {{
            flex-grow: 1;
            height: 4px;
            border-radius: 2px;
            background: #30363d;
        }}

        midi-visualizer {{
            display: block;
            width: 100%;
            background: #0d1117;
            border-radius: 8px;
            border: 1px solid #30363d;
            overflow: hidden;
        }}

        midi-visualizer .piano-roll-visualizer {{
            background: #0d1117;
            width: 100%;
            height: 100%;
            border-radius: 8px;
        }}

        midi-visualizer svg rect.note {{
            fill: #58a6ff;
            opacity: 0.75;
            stroke: #30363d;
            stroke-width: 0.5;
        }}

        midi-visualizer svg rect.note.active {{
            fill: #ff4b4b;
            opacity: 1;
            stroke: #e6edf3;
            stroke-width: 1;
        }}

        midi-visualizer svg rect.note:hover {{
            opacity: 1;
            stroke-width: 1.5;
            stroke: #79c0ff;
        }}

        .midi-container {{
            width: 100%;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        }}
    </style>

    <div class="midi-container">
        <midi-player src="{data_url}" sound-font visualizer="#pianoRollViz"></midi-player>
        <midi-visualizer src="{data_url}" type="piano-roll" id="pianoRollViz"></midi-visualizer>
    </div>
    """

    streamlit.components.v1.html(html_string, height=height, scrolling=False)


def render_audio_player(audio_bytes: bytes, label: str = "Audio") -> None:
    """
    Render a simple audio player for WAV files.

    Args:
        audio_bytes: Raw audio file bytes (WAV format)
        label: Label for the audio player (unused but kept for API consistency)
    """
    st.audio(audio_bytes, format="audio/wav")
