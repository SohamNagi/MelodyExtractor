# 🎵 MelodyExtractor

An interactive Streamlit application that extracts predominant melodies from audio files and converts them to MIDI. Upload a song, isolate vocals or instruments with AI-powered source separation, track the melody pitch, and download a clean MIDI file — all from your browser.

## Features

- **Audio Upload** — Drag-and-drop audio files (WAV, MP3, FLAC, OGG; up to 200 MB).
- **Source Separation** — Optionally isolate vocals or melodic stems using [Demucs](https://github.com/facebookresearch/demucs) before extraction.
- **Multiple Pitch Extractors** — Choose from three algorithms:
  | Extractor | Backend | Notes |
  |-----------|---------|-------|
  | **librosa pYIN** | librosa | Default; fast probabilistic YIN pitch tracker |
  | **CREPE** | torchcrepe | Neural pitch estimator; benefits from GPU |
  | **Essentia Melodia** | essentia | Predominant melody in polyphonic mixes |
- **Post-Processing Pipeline** — Confidence thresholding, median smoothing, pitch quantization (semitone / quarter-tone), minimum note length filtering, and optional note joining.
- **MIDI Generation & Playback** — Convert extracted notes to MIDI, preview in-browser with an embedded MIDI player, and download the `.mid` file.
- **Key Detection** — Estimate the musical key and scale via Essentia or librosa.
- **GPU Acceleration** — Automatic device selection (CUDA → MPS → CPU) for Demucs and CREPE.
- **Smart Caching** — SHA-256-based caching of separation and extraction results to avoid redundant computation.

## Requirements

- Python 3.10+

## Installation

```bash
# Clone the repository
git clone https://github.com/SohamNagi/MelodyExtractor.git
cd MelodyExtractor

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# (Optional) Install extra extractors & key detection
pip install -r requirements-optional.txt
```

The optional packages (`essentia`, `torchcrepe`) enable the Essentia Melodia and CREPE extractors respectively. The app works without them — unavailable extractors are simply hidden from the UI.

## Usage

```bash
streamlit run app.py
```

The app opens in your default browser. From the sidebar you can:

1. **Upload** an audio file.
2. **Configure separation** — enable/disable Demucs and choose a separation model (`htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra`).
3. **Select an extractor** and tune its parameters.
4. **Adjust post-processing** — confidence threshold, smoothing window, quantization mode, minimum note length, velocity method, and note joining.
5. **Run the pipeline** — the app walks through separation → extraction → post-processing → MIDI generation → key detection.
6. **Preview & download** — listen to the synthesized MIDI audio, view the MIDI in an embedded player, and download the `.mid` file.

## Project Structure

```
MelodyExtractor/
├── app.py                        # Streamlit UI orchestration
├── melody_extractor/
│   ├── __init__.py               # Package metadata
│   ├── extractors/
│   │   ├── __init__.py           # Extractor registry
│   │   ├── base.py               # Abstract MelodyExtractor base class
│   │   ├── pyin.py               # librosa pYIN extractor
│   │   ├── crepe.py              # CREPE neural extractor (optional)
│   │   └── melodia.py            # Essentia Melodia extractor (optional)
│   ├── separation.py             # Demucs source separation
│   ├── postprocessing.py         # Pitch processing & note segmentation
│   ├── midi_gen.py               # MIDI creation & synthesis
│   ├── key_detection.py          # Musical key estimation
│   ├── torch_backend.py          # Device selection (CUDA/MPS/CPU)
│   ├── visualizer.py             # In-browser MIDI player
│   └── utils.py                  # Audio I/O, hashing, tempo estimation
├── .streamlit/
│   └── config.toml               # Streamlit server & theme settings
├── requirements.txt              # Core dependencies
└── requirements-optional.txt     # Optional extractor dependencies
```

## Pipeline Overview

```
Audio File
  │
  ▼
Source Separation (Demucs)          ← optional
  │
  ▼
Melody Extraction (pYIN / CREPE / Melodia)
  │
  ▼
Post-Processing
  • confidence threshold
  • median smoothing
  • Hz → MIDI pitch conversion
  • quantization (semitone / quarter-tone)
  • note segmentation & velocity assignment
  • minimum length filter & note joining
  │
  ▼
MIDI Generation (PrettyMIDI)
  │
  ▼
Key Detection                      ← optional
```

## Configuration

Streamlit settings live in `.streamlit/config.toml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `maxUploadSize` | 200 MB | Maximum file upload size |
| `fileWatcherType` | `none` | Disables the file watcher for stability |
| `theme.base` | `dark` | Default UI theme |

## Adding a Custom Extractor

1. Create a new module under `melody_extractor/extractors/`.
2. Subclass `MelodyExtractor` from `melody_extractor/extractors/base.py`.
3. Implement the `extract(audio, sr, **kwargs)` method returning `(times, f0_hz, confidence)` NumPy arrays. Unvoiced frames should use `NaN` in `f0_hz`.
4. Set the class attributes `name` and `available`.
5. Register the class in `melody_extractor/extractors/__init__.py` (wrap the import in `try/except ImportError` if the dependency is optional).

## Dependencies

### Core (`requirements.txt`)

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.46.0 |
| numpy | ≥ 1.23, < 2.0 |
| librosa | ≥ 0.10, < 1.0 |
| soundfile | ≥ 0.12 |
| pretty-midi | ≥ 0.2.10 |
| demucs | ≥ 4.0 |
| torch | ≥ 2.0 |
| torchaudio | ≥ 2.0 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.10 |

### Optional (`requirements-optional.txt`)

| Package | Purpose |
|---------|---------|
| essentia | Melodia extractor and Essentia-based key detection |
| torchcrepe | CREPE neural pitch extractor |
