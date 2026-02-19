from melody_extractor.visualizer import render_midi_player
from melody_extractor.torch_backend import get_torch_runtime_info, select_torch_device
from melody_extractor.key_detection import detect_key, get_available_methods
from melody_extractor.midi_gen import notes_to_midi, midi_to_bytes, notes_to_dataframe, f0_to_dataframe
from melody_extractor.postprocessing import postprocess_pipeline
from melody_extractor.extractors import get_available_extractors, get_extractor
from melody_extractor.separation import separate_audio, get_available_models, DEMUCS_AVAILABLE
from melody_extractor.utils import compute_file_hash, load_audio, save_audio, estimate_tempo_bpm
from numpy.typing import NDArray
import numpy as np
import streamlit as st
import os
import tempfile
from pathlib import Path

_ = os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")


st.set_page_config(
    page_title="MelodyExtractor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state() -> None:
    state_keys: dict[str, object | None] = {
        "audio_bytes": None,
        "audio_path": None,
        "file_hash": None,
        "sep_result": None,
        "audio_for_extraction": None,
        "separation_device": None,
        "extractor_name": None,
        "times": None,
        "f0": None,
        "confidence": None,
        "notes": None,
        "midi_obj": None,
        "midi_bytes": None,
        "midi_wav_bytes": None,
        "source_bpm": None,
        "source_duration_sec": None,
        "key_result": None,
        "pipeline_statuses": {},
    }
    for key, default in state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default


@st.cache_data(show_spinner=False)
def cached_separate(
    audio_path: str,
    model_name: str,
    _file_hash: str,
    device: str,
) -> dict[str, str | bool | None]:
    return separate_audio(audio_path, model_name=model_name, device=device, cache_dir="./cache")


@st.cache_data(show_spinner=False)
def cached_source_tempo_bpm(audio_path: str, _file_hash: str) -> float:
    audio, sr = load_audio(audio_path)
    return estimate_tempo_bpm(audio, sr)


def set_pipeline_status(stage: str, label: str, state: str) -> None:
    current_statuses = st.session_state.get("pipeline_statuses")
    if not isinstance(current_statuses, dict):
        current_statuses = {}
    current_statuses[stage] = {"label": label, "state": state}
    st.session_state["pipeline_statuses"] = current_statuses


def render_persisted_pipeline_statuses(status_container) -> None:
    statuses = st.session_state.get("pipeline_statuses")
    if not isinstance(statuses, dict) or len(statuses) == 0:
        return

    with status_container:
        for stage in ["separation", "extraction", "postprocessing", "midi", "key_detection"]:
            status_info = statuses.get(stage)
            if not isinstance(status_info, dict):
                continue
            label = str(status_info.get("label", ""))
            state = str(status_info.get("state", "")).lower()
            if state == "complete":
                st.success(label)
            elif state == "error":
                st.error(label)
            elif state == "warning":
                st.warning(label)
            else:
                st.info(label)


def run_separation(
    audio_path: str,
    model_name: str,
    file_hash: str,
    status_container,
) -> dict[str, str | bool | None] | None:
    resolved_device = select_torch_device("auto")
    st.session_state["separation_device"] = resolved_device
    with status_container:
        with st.status("Running source separation...", expanded=True) as status:
            st.write(f"Using model: **{model_name}**")
            st.write(f"Compute backend: **{resolved_device}**")
            try:
                result = cached_separate(
                    audio_path, model_name, file_hash, resolved_device)
                if "error" in result:
                    st.warning(
                        f"Separation failed: {result['error']}. Melody extraction requires a successful Demucs separation.")
                    st.session_state["sep_result"] = None
                    st.session_state["audio_for_extraction"] = None
                    status.update(
                        label="Separation failed — extraction blocked", state="error")
                    set_pipeline_status(
                        "separation", "Separation failed — extraction blocked", "error")
                    return None
                st.session_state["sep_result"] = result
                extraction_audio_path = result.get("melody_path")
                if not isinstance(extraction_audio_path, str):
                    st.warning(
                        "Separation completed without a melody-isolated output. Melody extraction requires Demucs melody output.")
                    st.session_state["sep_result"] = None
                    st.session_state["audio_for_extraction"] = None
                    status.update(
                        label="Separation incomplete — extraction blocked", state="error")
                    set_pipeline_status(
                        "separation", "Separation incomplete — extraction blocked", "error")
                    return None

                st.session_state["audio_for_extraction"] = extraction_audio_path
                cached_label = " (cached)" if result.get("cached") else ""
                status.update(
                    label=f"Separation complete{cached_label}", state="complete")
                set_pipeline_status(
                    "separation", f"Separation complete{cached_label}", "complete")
                return result
            except Exception as e:
                st.warning(
                    f"Separation failed: {e}. Melody extraction requires a successful Demucs separation.")
                st.session_state["sep_result"] = None
                st.session_state["audio_for_extraction"] = None
                status.update(
                    label="Separation failed — extraction blocked", state="error")
                set_pipeline_status(
                    "separation", "Separation failed — extraction blocked", "error")
                return None


def render_pipeline_snapshot(audio_path: str | None) -> None:
    if not audio_path:
        st.info("Upload an audio file to get started.")
        return

    st.success("Audio loaded and ready.")

    sep_result = st.session_state.get("sep_result")
    if sep_result and "instrumental_path" in sep_result:
        cached_label = " (cached)" if sep_result.get("cached") else ""
        separation_device = st.session_state.get("separation_device")
        if separation_device:
            st.success(
                f"Source separation complete{cached_label} on {separation_device}.")
        else:
            st.success(f"Source separation complete{cached_label}.")

        instrumental_path = sep_result["instrumental_path"]
        if Path(instrumental_path).exists():
            st.caption("Isolated instrumental preview")
            st.audio(instrumental_path, format="audio/wav")

        melody_path = sep_result.get("melody_path")
        if isinstance(melody_path, str) and Path(melody_path).exists():
            st.caption("Melody-focused preview (drum-suppressed)")
            st.audio(melody_path, format="audio/wav")

    times = st.session_state.get("times")
    f0 = st.session_state.get("f0")
    confidence = st.session_state.get("confidence")
    if times is not None and f0 is not None and confidence is not None:
        extractor_name = st.session_state.get(
            "extractor_name") or "selected extractor"
        voiced_frames = int(np.sum(confidence > 0.0))
        st.success(
            f"Melody extraction complete with {extractor_name} ({len(times)} frames, {voiced_frames} voiced).")

    notes = st.session_state.get("notes")
    if notes is not None:
        if len(notes) > 0:
            st.success(
                f"Post-processing complete with {len(notes)} note events.")
        else:
            st.warning("Post-processing completed but no notes were detected.")

    midi_bytes = st.session_state.get("midi_bytes")
    if midi_bytes:
        source_bpm = st.session_state.get("source_bpm")
        if isinstance(source_bpm, (int, float)):
            st.success(
                f"MIDI generated ({len(midi_bytes):,} bytes). Detected source tempo: {float(source_bpm):.2f} BPM.")
        else:
            st.success(f"MIDI generated ({len(midi_bytes):,} bytes).")

    key_result = st.session_state.get("key_result")
    if key_result and "error" not in key_result:
        st.success(
            f"Key detection complete: {key_result.get('key', 'Unknown')} {key_result.get('scale', '')}".strip(
            )
        )


def run_extraction(
    audio_path: str,
    file_hash: str,
    extractor_name: str,
    extractor_params: dict[str, object],
    status_container,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    with status_container:
        with st.status("Extracting melody...", expanded=True) as status:
            st.write(f"Using extractor: **{extractor_name}**")
            try:
                audio, sr = load_audio(audio_path)
                extractor = get_extractor(extractor_name)
                times, f0, confidence = extractor.extract(
                    audio, sr, **extractor_params)
                st.session_state["times"] = times
                st.session_state["f0"] = f0
                st.session_state["confidence"] = confidence
                st.session_state["extractor_name"] = extractor_name
                valid_frames = np.sum(confidence > 0.0)
                st.write(
                    f"Extracted {len(times)} frames, {valid_frames} with non-zero confidence")
                status.update(label="Melody extraction complete",
                              state="complete")
                set_pipeline_status(
                    "extraction", "Melody extraction complete", "complete")
                return times, f0, confidence
            except Exception as e:
                if extractor_name == "CREPE (torchcrepe)":
                    st.warning(
                        f"CREPE failed ({e}). Falling back to librosa pYIN.")
                    try:
                        audio, sr = load_audio(audio_path)
                        fallback_extractor = get_extractor("librosa pYIN")
                        times, f0, confidence = fallback_extractor.extract(
                            audio, sr)
                        st.session_state["times"] = times
                        st.session_state["f0"] = f0
                        st.session_state["confidence"] = confidence
                        st.session_state["extractor_name"] = "librosa pYIN (fallback)"
                        valid_frames = np.sum(confidence > 0.0)
                        st.write(
                            f"Fallback extracted {len(times)} frames, {valid_frames} with non-zero confidence")
                        status.update(
                            label="Melody extraction complete (fallback)", state="complete")
                        set_pipeline_status(
                            "extraction", "Melody extraction complete (fallback)", "complete")
                        return times, f0, confidence
                    except Exception as fallback_exc:
                        st.error(
                            f"Extraction failed: {e}. Fallback failed: {fallback_exc}")
                        status.update(label="Extraction failed", state="error")
                        set_pipeline_status(
                            "extraction", "Extraction failed", "error")
                        return None

                st.error(f"Extraction failed: {e}")
                status.update(label="Extraction failed", state="error")
                set_pipeline_status("extraction", "Extraction failed", "error")
                return None


def run_postprocessing(
    times: NDArray[np.float64],
    f0: NDArray[np.float64],
    confidence: NDArray[np.float64],
    confidence_threshold: float,
    smoothing_window: int,
    quantize: str,
    min_note_length: float,
    velocity_method: str,
    join_notes: bool,
    status_container,
) -> list[dict[str, object]] | None:
    with status_container:
        with st.status("Post-processing notes...", expanded=True) as status:
            try:
                notes = postprocess_pipeline(
                    times,
                    f0,
                    confidence,
                    confidence_threshold=confidence_threshold,
                    smoothing_window=smoothing_window,
                    quantize=quantize,
                    min_note_length=min_note_length,
                    velocity_method=velocity_method,
                    join_notes=join_notes,
                )
                st.session_state["notes"] = notes
                if not notes:
                    st.warning(
                        "No notes detected. Try switching the extraction method, "
                        "lowering the confidence threshold, or enabling source separation."
                    )
                    status.update(
                        label="Post-processing complete — no notes detected", state="error")
                    set_pipeline_status(
                        "postprocessing", "Post-processing complete — no notes detected", "error")
                    return notes
                st.write(f"Detected **{len(notes)}** note events")
                status.update(
                    label=f"Post-processing complete — {len(notes)} notes", state="complete")
                set_pipeline_status(
                    "postprocessing", f"Post-processing complete — {len(notes)} notes", "complete")
                return notes
            except Exception as e:
                st.error(f"Post-processing failed: {e}")
                status.update(label="Post-processing failed", state="error")
                set_pipeline_status(
                    "postprocessing", "Post-processing failed", "error")
                return None


def run_midi_generation(
    notes: list[dict[str, object]],
    source_bpm: float | None,
    status_container,
) -> bytes | None:
    with status_container:
        with st.status("Generating MIDI...", expanded=True) as status:
            try:
                tempo_bpm = float(source_bpm) if isinstance(
                    source_bpm, (int, float)) else 120.0
                midi_obj = notes_to_midi(notes, tempo=tempo_bpm, program=0)
                midi_bytes = midi_to_bytes(midi_obj)
                st.session_state["midi_obj"] = midi_obj
                st.session_state["midi_bytes"] = midi_bytes
                st.session_state["midi_wav_bytes"] = None
                if isinstance(source_bpm, (int, float)):
                    st.write(
                        f"Detected source tempo: **{float(source_bpm):.2f} BPM** (MIDI timing preserved from note timestamps)"
                    )
                st.write(f"Generated MIDI: **{len(midi_bytes):,}** bytes")
                status.update(label="MIDI generation complete",
                              state="complete")
                set_pipeline_status(
                    "midi", "MIDI generation complete", "complete")
                return midi_bytes
            except Exception as e:
                st.error(f"MIDI generation failed: {e}")
                status.update(label="MIDI generation failed", state="error")
                set_pipeline_status("midi", "MIDI generation failed", "error")
                return None


def run_key_detection(audio_path: str, method: str, status_container) -> dict[str, object] | None:
    with status_container:
        with st.status("Detecting key/scale...", expanded=True) as status:
            try:
                audio, sr = load_audio(audio_path)
                result = detect_key(audio, sr=sr, method=method)
                st.session_state["key_result"] = result
                if "error" in result:
                    st.warning(f"Key detection warning: {result['error']}")
                    status.update(
                        label="Key detection completed with warnings", state="error")
                    set_pipeline_status(
                        "key_detection", "Key detection completed with warnings", "error")
                else:
                    st.write(
                        f"Detected key: **{result['key']} {result['scale']}** (confidence: {result['confidence']:.2f})")
                    status.update(label="Key detection complete",
                                  state="complete")
                    set_pipeline_status(
                        "key_detection", "Key detection complete", "complete")
                return result
            except Exception as e:
                st.error(f"Key detection failed: {e}")
                status.update(label="Key detection failed", state="error")
                set_pipeline_status(
                    "key_detection", "Key detection failed", "error")
                return None


def save_uploaded_audio(uploaded_file) -> tuple[str, str] | None:
    try:
        raw_bytes = uploaded_file.read()
        file_hash = compute_file_hash(raw_bytes)

        if st.session_state.get("file_hash") == file_hash and st.session_state.get("audio_path"):
            existing_path = st.session_state["audio_path"]
            if Path(existing_path).exists():
                if st.session_state.get("source_bpm") is None:
                    st.session_state["source_bpm"] = cached_source_tempo_bpm(
                        existing_path, file_hash)
                if st.session_state.get("source_duration_sec") is None:
                    cached_audio, cached_sr = load_audio(existing_path)
                    st.session_state["source_duration_sec"] = float(
                        len(cached_audio) / float(cached_sr))
                return existing_path, file_hash

        audio, sr = load_audio(raw_bytes)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()
        save_audio(audio, sr, tmp_path)

        st.session_state["audio_bytes"] = raw_bytes
        st.session_state["audio_path"] = tmp_path
        st.session_state["file_hash"] = file_hash
        st.session_state["sep_result"] = None
        st.session_state["audio_for_extraction"] = None
        st.session_state["separation_device"] = None
        st.session_state["extractor_name"] = None
        st.session_state["times"] = None
        st.session_state["f0"] = None
        st.session_state["confidence"] = None
        st.session_state["notes"] = None
        st.session_state["midi_obj"] = None
        st.session_state["midi_bytes"] = None
        st.session_state["midi_wav_bytes"] = None
        st.session_state["source_bpm"] = cached_source_tempo_bpm(
            tmp_path, file_hash)
        st.session_state["source_duration_sec"] = float(len(audio) / float(sr))
        st.session_state["key_result"] = None
        st.session_state["pipeline_statuses"] = {}

        return tmp_path, file_hash
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        return None


def main() -> None:
    init_session_state()

    st.title("🎵 MelodyExtractor")
    st.caption(
        "Extract melodies from audio and generate MIDI — powered by Demucs & pYIN")

    with st.sidebar:
        st.header("Controls")

        uploaded_file = st.file_uploader(
            "Upload Audio",
            type=["mp3", "wav", "m4a", "flac"],
            help="Max 200MB. Supported formats: MP3, WAV, M4A, FLAC.",
        )

        if uploaded_file is not None:
            st.audio(uploaded_file,
                     format=f"audio/{uploaded_file.name.split('.')[-1]}")
            result = save_uploaded_audio(uploaded_file)
            if result:
                audio_path, file_hash = result
            else:
                audio_path = None
                file_hash = ""
        else:
            audio_path = st.session_state.get("audio_path")
            file_hash: str = st.session_state.get("file_hash") or ""

        if audio_path and file_hash and st.session_state.get("source_bpm") is None:
            st.session_state["source_bpm"] = cached_source_tempo_bpm(
                audio_path, file_hash)

        source_bpm = st.session_state.get("source_bpm")
        if isinstance(source_bpm, (int, float)):
            st.caption(f"Detected source tempo: {float(source_bpm):.2f} BPM")

        st.divider()

        with st.expander("Source Separation", expanded=True):
            separation_available = DEMUCS_AVAILABLE
            enable_separation = st.checkbox(
                "Enable Separation",
                value=True,
                disabled=not separation_available,
                help="Requires Demucs. Separates vocals from instrumental before extraction."
                if separation_available
                else "Demucs is not installed. Install it to enable source separation.",
            )
            torch_runtime = get_torch_runtime_info()
            if separation_available and torch_runtime["torch_available"]:
                if torch_runtime["mps_available"]:
                    st.caption(
                        "Acceleration: mps (Metal) is available and will be used on macOS.")
                elif torch_runtime["cuda_available"]:
                    st.caption(
                        "Acceleration: CUDA is available and will be used.")
                else:
                    st.caption(
                        "Acceleration: running on CPU (MPS/CUDA unavailable).")

            available_models = get_available_models() or ["htdemucs"]
            if enable_separation:
                sep_model: str = st.selectbox(
                    "Model",
                    options=available_models,
                    index=0,
                    help="Demucs model to use for source separation.",
                ) or available_models[0]
            else:
                sep_model = available_models[0]

        with st.expander("Melody Extraction", expanded=True):
            available_extractors = get_available_extractors()
            extractor_names = list(available_extractors.keys(
            )) if available_extractors else ["librosa pYIN"]
            selected_extractor_name: str = st.selectbox(
                "Algorithm",
                options=extractor_names,
                index=0,
                help="Melody extraction algorithm.",
            ) or extractor_names[0]

        with st.expander("Advanced Parameters", expanded=False):
            extractor_params: dict[str, object] = {}
            try:
                extractor_instance = get_extractor(selected_extractor_name)
                default_params = extractor_instance.get_default_params()
                param_descriptions = extractor_instance.get_param_descriptions()
                for param_name, default_value in default_params.items():
                    description = param_descriptions.get(param_name, "")
                    if isinstance(default_value, (int, float)):
                        extractor_params[param_name] = st.number_input(
                            param_name,
                            value=default_value,
                            help=description if description else None,
                        )
                    elif isinstance(default_value, str):
                        extractor_params[param_name] = st.text_input(
                            param_name,
                            value=default_value,
                            help=description if description else None,
                        )
                    else:
                        extractor_params[param_name] = default_value
            except Exception:
                extractor_params = {}

        with st.expander("Post-Processing", expanded=False):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Minimum confidence to keep a frequency estimate.",
            )
            smoothing_window = st.slider(
                "Smoothing Window",
                min_value=1,
                max_value=15,
                value=5,
                step=2,
                help="Median filter window for smoothing the F0 contour.",
            )
            quantize: str = st.selectbox(
                "Quantization",
                options=["semitone", "quarter", "none"],
                index=0,
                help="Pitch quantization resolution.",
            ) or "semitone"
            min_note_length = st.slider(
                "Min Note Length (s)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Minimum duration for a note event in seconds.",
            )
            velocity_method: str = st.selectbox(
                "Velocity Method",
                options=["from_confidence", "fixed"],
                index=0,
                help="How to compute MIDI velocity for each note.",
            ) or "from_confidence"
            join_notes = st.toggle(
                "Join Notes (Legato)",
                value=False,
                help="Extend each note to the next note start to reduce gaps between notes.",
            )

        with st.expander("Key Detection", expanded=False):
            enable_key_detection = st.checkbox(
                "Enable Key Detection", value=True)
            available_key_methods = get_available_methods() or ["auto"]
            if enable_key_detection:
                key_method: str = st.selectbox(
                    "Method",
                    options=available_key_methods,
                    index=0,
                    help="Key detection algorithm.",
                ) or available_key_methods[0]
            else:
                key_method = available_key_methods[0]

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            btn_separation = st.button(
                "Run Separation", use_container_width=True)
            btn_midi = st.button("Generate MIDI", use_container_width=True)
        with col_b:
            btn_extract = st.button("Extract Melody", use_container_width=True)
            btn_all = st.button("Run All", type="primary",
                                use_container_width=True)

    tab_pipeline, tab_midi, tab_analysis, tab_downloads = st.tabs(
        ["Pipeline", "MIDI Player", "Analysis", "Downloads"]
    )

    pipeline_snapshot_placeholder = tab_pipeline.empty()
    pipeline_status_placeholder = tab_pipeline.empty()

    with pipeline_snapshot_placeholder.container():
        render_pipeline_snapshot(audio_path)
    render_persisted_pipeline_statuses(pipeline_status_placeholder)

    if btn_separation:
        if not audio_path:
            st.sidebar.error("Upload a file first.")
        elif not enable_separation:
            st.sidebar.warning("Source separation is disabled.")
        else:
            with tab_pipeline:
                run_separation(audio_path, sep_model, file_hash,
                               pipeline_status_placeholder)

    if btn_extract:
        if not audio_path:
            st.sidebar.error("Upload a file first.")
        elif not enable_separation:
            st.sidebar.error(
                "Melody extraction requires Demucs separation to generate the melody-isolated instrumental.")
        else:
            source_for_extraction = st.session_state.get(
                "audio_for_extraction")
            if not isinstance(source_for_extraction, str):
                st.sidebar.error(
                    "Run separation first. Extraction is restricted to Demucs melody output only.")
            else:
                with tab_pipeline:
                    result = run_extraction(
                        source_for_extraction,
                        file_hash,
                        selected_extractor_name,
                        extractor_params,
                        pipeline_status_placeholder,
                    )
                    if result is not None:
                        times, f0, confidence = result
                        run_postprocessing(
                            times,
                            f0,
                            confidence,
                            confidence_threshold,
                            smoothing_window,
                            quantize,
                            min_note_length,
                            velocity_method,
                            join_notes,
                            pipeline_status_placeholder,
                        )

    if btn_midi:
        notes = st.session_state.get("notes")
        source_bpm = st.session_state.get("source_bpm")
        if notes is None:
            st.sidebar.error("Run extraction first to generate notes.")
        elif len(notes) == 0:
            st.sidebar.warning("No notes detected — cannot generate MIDI.")
        else:
            with tab_pipeline:
                run_midi_generation(
                    notes, float(source_bpm) if isinstance(source_bpm, (int, float)) else None, pipeline_status_placeholder)

    if btn_all:
        if not audio_path:
            st.sidebar.error("Upload a file first.")
        elif not enable_separation:
            st.sidebar.error(
                "Run All requires Demucs separation so extraction can use the melody-isolated instrumental.")
        else:
            with tab_pipeline:
                current_audio_path = audio_path

                sep_result = run_separation(
                    audio_path, sep_model, file_hash, pipeline_status_placeholder)
                if sep_result and "melody_path" in sep_result:
                    candidate_path = sep_result.get("melody_path")
                    if isinstance(candidate_path, str):
                        current_audio_path = candidate_path
                    else:
                        st.sidebar.error(
                            "Demucs did not produce a melody output. Extraction was not run.")
                        current_audio_path = ""
                else:
                    st.sidebar.error(
                        "Separation failed. Extraction was not run because only Demucs melody output is allowed.")
                    current_audio_path = ""

                extraction_result = None
                if current_audio_path:
                    extraction_result = run_extraction(
                        current_audio_path,
                        file_hash,
                        selected_extractor_name,
                        extractor_params,
                        pipeline_status_placeholder,
                    )

                if extraction_result is not None:
                    times, f0, confidence = extraction_result
                    notes = run_postprocessing(
                        times,
                        f0,
                        confidence,
                        confidence_threshold,
                        smoothing_window,
                        quantize,
                        min_note_length,
                        velocity_method,
                        join_notes,
                        pipeline_status_placeholder,
                    )

                    if notes and len(notes) > 0:
                        source_bpm = st.session_state.get("source_bpm")
                        run_midi_generation(
                            notes,
                            float(source_bpm) if isinstance(
                                source_bpm, (int, float)) else None,
                            pipeline_status_placeholder,
                        )

                        if enable_key_detection:
                            key_audio = current_audio_path
                            run_key_detection(
                                key_audio, key_method, pipeline_status_placeholder)

    with pipeline_snapshot_placeholder.container():
        render_pipeline_snapshot(audio_path)
    render_persisted_pipeline_statuses(pipeline_status_placeholder)

    with tab_midi:
        midi_bytes = st.session_state.get("midi_bytes")
        if midi_bytes:
            source_duration_sec = st.session_state.get("source_duration_sec")
            render_midi_player(
                midi_bytes,
                height=500,
                target_duration_sec=float(source_duration_sec)
                if isinstance(source_duration_sec, (int, float))
                else None,
            )
        else:
            st.info(
                "No MIDI generated yet. Run extraction and MIDI generation first.")

    with tab_analysis:
        key_result = st.session_state.get("key_result")
        if key_result:
            st.subheader("Key Detection")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Key", key_result.get("key", "—"))
            m2.metric("Scale", key_result.get("scale", "—"))
            m3.metric("Confidence", f"{key_result.get('confidence', 0.0):.2f}")
            m4.metric("Method", key_result.get("method", "—"))

        notes = st.session_state.get("notes")
        if notes is not None:
            st.subheader("Note Events")
            if len(notes) > 0:
                df_notes = notes_to_dataframe(notes)
                st.dataframe(df_notes, use_container_width=True)
            else:
                st.warning(
                    "No notes detected. Try switching the extraction method, "
                    "lowering the confidence threshold, or enabling source separation."
                )

        times = st.session_state.get("times")
        f0 = st.session_state.get("f0")
        confidence_arr = st.session_state.get("confidence")
        if times is not None and f0 is not None and confidence_arr is not None:
            with st.expander("F0 Contour"):
                df_f0 = f0_to_dataframe(times, f0, confidence_arr)
                st.dataframe(df_f0, use_container_width=True)

        if notes is None and times is None:
            st.info("No analysis data yet. Run melody extraction first.")

    with tab_downloads:
        midi_bytes = st.session_state.get("midi_bytes")
        if midi_bytes:
            st.download_button(
                label="Download MIDI",
                data=midi_bytes,
                file_name="melody.mid",
                mime="audio/midi",
                use_container_width=True,
            )
        else:
            st.button("Download MIDI", disabled=True,
                      use_container_width=True, help="Generate MIDI first.")

        notes = st.session_state.get("notes")
        if notes and len(notes) > 0:
            df_notes = notes_to_dataframe(notes)
            csv_bytes = df_notes.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Notes CSV",
                data=csv_bytes,
                file_name="notes.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Download Notes CSV", disabled=True,
                      use_container_width=True, help="Run extraction first.")

        sep_result = st.session_state.get("sep_result")
        if sep_result and "melody_path" in sep_result:
            melody_path = sep_result["melody_path"]
            if isinstance(melody_path, str) and Path(melody_path).exists():
                with open(melody_path, "rb") as f:
                    melody_bytes = f.read()
                st.download_button(
                    label="Download Melody-Focused WAV",
                    data=melody_bytes,
                    file_name="melody_focused.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )

        if sep_result and "instrumental_path" in sep_result:
            instrumental_path = sep_result["instrumental_path"]
            if Path(instrumental_path).exists():
                with open(instrumental_path, "rb") as f:
                    instrumental_bytes = f.read()
                st.download_button(
                    label="Download Instrumental WAV",
                    data=instrumental_bytes,
                    file_name="instrumental.wav",
                    mime="audio/wav",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
