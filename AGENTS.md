# Project Guidelines

## Code Style
- Target Python 3.10+ and use modern typing (`X | Y`, `dict[str, ...]`, `list[dict]`) as in `app.py`, `melody_extractor/utils.py`, and `melody_extractor/postprocessing.py`.
- Keep function signatures and return types annotated across backend modules, matching `melody_extractor/extractors/base.py`.
- Preserve existing docstring style (concise Google-style Args/Returns in backend functions).
- Do not add `TODO`/`FIXME`/`HACK` comments.

## Architecture
- `app.py` is the single Streamlit orchestration layer (state, controls, and stage execution).
- Backend processing stays in `melody_extractor/` (no Streamlit imports there except `melody_extractor/visualizer.py`).
- Pipeline flow is: separation -> extraction -> postprocess -> MIDI generation -> optional key detection.
- Maintain extractor contract from `melody_extractor/extractors/base.py` and registry behavior in `melody_extractor/extractors/__init__.py`.

## Build and Test
- Install core dependencies: `pip install -r requirements.txt`
- Install optional extractors/key tooling: `pip install -r requirements-optional.txt`
- Run app locally: `streamlit run app.py`
- No formal automated test suite is currently present; validate changes by running the Streamlit app and the affected pipeline stage(s).

## Project Conventions
- Optional dependencies (Demucs, Essentia, torchcrepe) must degrade gracefully via `try/except ImportError` and availability flags.
- Unvoiced pitch frames are normalized to `NaN` across extractors; preserve this behavior when editing extraction/postprocessing.
- Keep Streamlit file-watcher suppression behavior aligned with `.streamlit/config.toml`, `app.py`, and `melody_extractor/torch_backend.py`.
- Separation caching is hash/model keyed under `cache/<sha256>/<model>/`; preserve deterministic cache keys.

## Integration Points
- Demucs integration uses `demucs.pretrained.get_model` + `demucs.apply.apply_model` in `melody_extractor/separation.py` (not `demucs.api`).
- Key detection supports `auto`, `essentia`, and `librosa` paths in `melody_extractor/key_detection.py`.
- MIDI playback/visualization relies on `html-midi-player` web components and CDN scripts in `melody_extractor/visualizer.py`.
- Runtime device selection for torch-backed operations is centralized in `melody_extractor/torch_backend.py`.

## Security
- Uploaded audio identity is SHA-256 based (`compute_file_hash`) and used for cache/session identity; keep this flow intact.
- Keep file I/O local (temp files + `cache/`) and avoid introducing remote upload/storage side effects.
- If changing `melody_extractor/visualizer.py`, review third-party CDN script usage and avoid adding unnecessary external script sources.
