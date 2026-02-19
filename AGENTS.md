# AGENTS.md — MelodyExtractor

## What This Project Is

A Streamlit web application that extracts the predominant melody from an audio file and produces a clean single-track MIDI. The user uploads a song, the app optionally separates it into stems, extracts the melody pitch contour, converts it to discrete note events, generates a MIDI file, and lets the user play/visualize the result in-browser via a piano roll.

## How to Run

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt  # for Essentia + torchcrepe extractors
streamlit run app.py
```

Python 3.10+ required (codebase uses `X | Y` union syntax, not `Union[X, Y]`).

## Project Structure

```
MelodyExtractor/
├── app.py                              # Streamlit UI — single-page app, all routing
├── requirements.txt                    # Core deps (streamlit, librosa, demucs, torch, etc.)
├── requirements-optional.txt           # Optional deps (essentia, torchcrepe)
├── .streamlit/config.toml              # Streamlit config (upload limits, watcher mode)
├── .gitignore
└── melody_extractor/                   # Backend package — no Streamlit imports here
    ├── __init__.py                     # Package marker, exports __version__
    ├── utils.py                        # Audio I/O, hashing, Hz/MIDI conversions
    ├── torch_backend.py                # Torch device selection + runtime capability helpers
    ├── separation.py                   # Demucs source separation with caching
    ├── postprocessing.py               # F0 contour → discrete note events pipeline
    ├── midi_gen.py                     # Note events → PrettyMIDI → bytes/file/DataFrame
    ├── key_detection.py                # Musical key estimation (Essentia or librosa)
    ├── visualizer.py                   # html-midi-player Streamlit component
    └── extractors/                     # Pluggable melody extraction algorithms
        ├── __init__.py                 # Registry: name → extractor class mapping
        ├── base.py                     # ABC defining the extractor interface
        ├── melodia.py                  # Essentia PredominantPitchMelodia (optional)
        ├── crepe.py                    # torchcrepe neural pitch tracker (optional)
        └── pyin.py                     # librosa pYIN (always available)
```

## Architecture

### Pipeline Flow

```
Upload (mp3/wav/m4a/flac)
  │
  ├─ [Optional] Source Separation (Demucs)
  │    └─ Produces instrumental WAV (drums + bass + other)
  │
  ├─ Melody Extraction (one of 3 algorithms)
  │    └─ Produces frame-level (times, f0_hz, confidence) arrays
  │
  ├─ Post-Processing
  │    └─ confidence threshold → median smooth → Hz→MIDI → quantize → segment → velocity
  │    └─ Produces list[dict] note events
  │
  ├─ MIDI Generation
  │    └─ Produces PrettyMIDI object + raw bytes
  │
  └─ [Optional] Key Detection
       └─ Produces {key, scale, confidence, method}
```

### Module Responsibilities

#### `app.py`
The Streamlit UI. Owns all state management, user controls, and pipeline orchestration.

- **Sidebar**: File upload + audio preview, separation controls, extractor selection + advanced params, post-processing sliders, key detection toggle, 4 action buttons (Run Separation / Extract Melody / Generate MIDI / Run All).
- **Main area**: 4 tabs — Pipeline (persistent step snapshots + status messages + isolated WAV preview), MIDI Player (html-midi-player), Analysis (key metrics + note table + F0 contour), Downloads (MIDI/CSV/WAV).
- **Session state**: Tracks all pipeline artifacts plus device/runtime metadata and persisted pipeline status per stage. Reset when a new file is uploaded.
- **Caching**: `@st.cache_data` on separation (keyed by file hash + model name + resolved device).

#### `melody_extractor/torch_backend.py`
Shared torch runtime helpers.

- `select_torch_device(requested="auto")` — Resolves best available backend (`mps` on macOS when available, otherwise `cuda`, then `cpu`).
- `get_torch_runtime_info()` — Reports backend capabilities (`cuda_available`, `mps_built`, `mps_available`) and selected runtime device.

#### `melody_extractor/utils.py` (184 lines)
Shared utilities. No Streamlit dependency.

- `compute_file_hash(bytes) → str` — SHA256 hex digest for cache keys.
- `load_audio(source, sr=44100)` — Loads from file path or raw bytes via librosa, returns mono float32.
- `save_audio(array, sr, path)` — Writes via soundfile.
- `hz_to_midi(freq)` — Vectorized Hz→MIDI conversion (handles 0/NaN/negative).
- `hash_params(**kwargs) → str` — Deterministic parameter hashing for cache discrimination.

#### `melody_extractor/separation.py`
Demucs v4.0.x integration. Gracefully degrades when Demucs is not installed (`DEMUCS_AVAILABLE` flag).

- Uses `demucs.pretrained.get_model()` + `demucs.apply.apply_model()` + `demucs.audio.AudioFile`.
- Builds instrumental by summing all sources except vocals.
- File-system cache: `./cache/{file_hash}/{model_name}/instrumental.wav`.
- Device selection comes from `torch_backend.select_torch_device` (`auto`, `cpu`, `cuda`, `mps`).
- Accelerator fallback: if non-CPU separation fails, retries on CPU.
- On any failure: returns original audio path + error message (pipeline continues).

#### `melody_extractor/extractors/` (5 files)
Pluggable extractor system with registry pattern.

**`base.py`** — Abstract base class `MelodyExtractor`:
- `extract(audio, sr, **kwargs) → (times, f0_hz, confidence)` — All extractors return this standardized tuple.
- `get_default_params() → dict` — Algorithm-specific defaults.
- `get_param_descriptions() → dict` — Human-readable help text per parameter.

**`pyin.py`** — Always available. Uses `librosa.pyin()`. NaN = unvoiced (native behavior).

**`melodia.py`** — Optional (requires `essentia`). Uses `EqualLoudness` + `PredominantPitchMelodia`. Converts Essentia's `0.0 = unvoiced` to NaN.

**`crepe.py`** — Optional (requires `torch` + `torchcrepe`). Input must be `(1, N)` float tensor on selected runtime device (`mps`/`cuda`/`cpu`). Applies median filter to periodicity, then `torchcrepe.threshold.At()` to silence unvoiced frames.

**`__init__.py`** — Registry. Imports pYIN unconditionally, tries melodia/crepe with `try/except ImportError`. Exports:
- `EXTRACTORS: dict[str, type]` — name → class for available extractors.
- `get_available_extractors()` — Returns copy of registry.
- `get_extractor(name)` — Instantiates by name, raises `KeyError` if unknown.

#### `melody_extractor/postprocessing.py` (289 lines)
Converts raw frame-level pitch data into discrete note events. Six-stage pipeline:

1. **Confidence threshold** — Frames below threshold → NaN (silenced).
2. **Smooth pitch** — Median filter on voiced regions, NaN gaps preserved.
3. **Hz → MIDI** — Vectorized log2 conversion, invalid values → NaN.
4. **Quantize** — Semitone (round), quarter-tone (0.5), or none.
5. **Segment** — Groups consecutive same-pitch frames into note events with start/end/pitch/velocity/confidence_avg.
6. **Velocity** — Either `from_confidence` (40 + conf * 80, clamped 1–127) or `fixed` (80).

Main entry point: `postprocess_pipeline(times, f0, confidence, ...) → list[dict]`.

#### `melody_extractor/midi_gen.py` (120 lines)
MIDI generation and data export.

- `notes_to_midi(notes, tempo=120, program=0) → PrettyMIDI` — Single-track, single-instrument.
- `midi_to_bytes(pm) → bytes` — In-memory serialization via BytesIO.
- `notes_to_dataframe(notes) → DataFrame` — Columns: start, end, duration, pitch, note_name, velocity, confidence.
- `f0_to_dataframe(times, f0, confidence) → DataFrame` — Raw F0 contour export.

#### `melody_extractor/key_detection.py` (135 lines)
Musical key and scale estimation.

- **Essentia path**: `es.KeyExtractor(sampleRate=sr)` → returns (key, scale, strength).
- **librosa path**: Computes chroma CQT, correlates against all 24 Krumhansl-Schmuckler key profiles (12 major + 12 minor), picks highest correlation.
- `detect_key(audio, sr, method="auto")` — "auto" tries Essentia first, falls back to librosa. Returns `{key, scale, confidence, method}`. On error returns `{key: "Unknown", error: ...}`.

#### `melody_extractor/visualizer.py` (147 lines)
In-browser MIDI playback via html-midi-player web components.

- Encodes MIDI bytes as base64 data URL.
- Renders `<midi-player>` + `<midi-visualizer type="piano-roll">` inside `st.components.v1.html()`.
- Custom CSS: dark theme matching Streamlit, red play button, blue notes with hover/active states.
- CDN: `tone@14.7.58` + `@magenta/music@1.23.1` + `html-midi-player@1.5.0`.

## Key Design Decisions

### Optional Dependencies
Essentia, torchcrepe, and Demucs are all optional. Each module wraps its import in `try/except` and exposes an `_AVAILABLE` flag. The UI disables controls when deps are missing. The only always-available extractor is librosa pYIN. This means the app runs with just `pip install -r requirements.txt`.

### Unvoiced Frame Convention
All extractors normalize to NaN = unvoiced. Essentia natively uses 0.0 (converted in melodia.py), librosa pYIN natively uses NaN, torchcrepe uses the `At` threshold which produces NaN. Post-processing relies on this NaN convention for gap detection.

### Caching Strategy
- **Separation**: File-system cache in `./cache/{sha256}/{model}/`. Checked before running Demucs. Also wrapped in `@st.cache_data` keyed on `(audio_path, model_name, file_hash)`.
- **Session state**: All pipeline artifacts stored in `st.session_state` so tabs can access results without re-running.
- **Pipeline status UI**: Stage status labels are persisted in session state and re-rendered on rerun, so changing widgets does not clear completed-step visibility.
- **File upload**: SHA256 hash of uploaded bytes used to detect re-uploads of the same file and skip reprocessing.

### Error Recovery
Every pipeline stage is wrapped in try/except. Separation failure falls back to original audio with a warning. Extraction failure stops the pipeline. Zero notes detected shows troubleshooting suggestions. Key detection failure returns a structured error dict.

## Conventions

- **Python version**: 3.10+. Use `X | Y` for unions, never `Union[X, Y]` or `Optional[X]`.
- **No Streamlit in backend**: The `melody_extractor/` package never imports Streamlit (except `visualizer.py` which is inherently a UI component).
- **Type annotations**: All function signatures are annotated. `np.ndarray` without generic params is acceptable.
- **Docstrings**: Google-style Args/Returns format throughout.
- **No TODOs**: The codebase should contain zero TODO/FIXME/HACK comments.

## Dependencies

### Core (`requirements.txt`)
| Package | Purpose |
|---------|---------|
| streamlit >= 1.46 | Web UI framework (includes torch watcher compatibility fix) |
| numpy >= 1.23 | Array operations |
| librosa >= 0.10 | Audio loading, pYIN, chroma |
| soundfile >= 0.12 | WAV writing |
| pretty-midi >= 0.2.10 | MIDI generation |
| demucs >= 4.0 | Source separation |
| torch >= 2.0 | Demucs backend |
| torchaudio >= 2.0 | Demucs audio loading |
| pandas >= 2.0 | DataFrames for note tables |
| scipy >= 1.10 | Median filter in post-processing |

### Optional (`requirements-optional.txt`)
| Package | Purpose |
|---------|---------|
| essentia | PredominantPitchMelodia extractor + KeyExtractor |
| torchcrepe | CREPE neural pitch tracker |

## Gotchas

- **Demucs v4.0.x has no `demucs.api` module.** The `Separator` class mentioned in some docs is from an unreleased version. We use `demucs.pretrained.get_model()` + `demucs.apply.apply_model()` instead.
- **librosa >= 0.10** requires keyword-only args for `pyin(fmin=, fmax=)`. Positional args will break.
- **Streamlit expanders cannot be nested.** The "Advanced Parameters" section is a sibling expander, not nested inside "Melody Extraction".
- **PyTorch + Streamlit watcher noise (`torch.classes`)** can appear on older Streamlit versions. Keep Streamlit on `>=1.46` and use `server.fileWatcherType = "none"` in this repo config.
- **html-midi-player** needs the `sound-font` attribute (even if empty) on `<midi-player>` to trigger the default hosted SoundFont for playback.
- **torchcrepe input shape** must be `(1, N)` float tensor — not `(N,)`.
- **Essentia PredominantPitchMelodia** returns `0.0` for unvoiced frames, not NaN. Must convert.
