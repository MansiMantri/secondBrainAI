"""
SecondBrainAI Local Version - Uses Ollama instead of Gemini API.

Run entirely locally with no API keys required.
"""

import time
from pathlib import Path

import streamlit as st

from ingest_logic import VIDEO_EXTS, extract_audio_from_video, media_to_images
from audio_engine import transcribe_audio
from vision_engine_local import (
    visual_summary_from_image,
    check_ollama_models,
    DEFAULT_VISION_MODEL,
    DEFAULT_TEXT_MODEL,
    format_multi_source_context,
    synthesize_multi_source_corpus,
    answer_prompt_over_corpus,
)


st.set_page_config(page_title="SecondBrainAI Local (Ollama)", layout="centered")
st.title("SecondBrainAI Vision mRAG - Local Version")

st.write(
    "Upload one or more PDFs, images, or videos. Uses local Ollama models for OCR-free "
    "visual summaries and combined reasoning."
)
st.write("**No API keys required** - runs entirely on your machine.")

models = check_ollama_models()

# Check Ollama status
with st.expander("🔍 Ollama Status"):
    if models.get("error"):
        st.error(f"Ollama connection failed: {models['error']}")
        st.info("Make sure Ollama is running: `ollama serve`")
    else:
        st.success("Ollama connected successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vision Models")
            if models["vision"]:
                for model in models["vision"]:
                    st.code(model)
            else:
                st.warning("No vision models found. Install with:")
                st.code("ollama pull llava:7b")

        with col2:
            st.subheader("Text Models")
            if models["text"]:
                for model in models["text"]:
                    st.code(model)
            else:
                st.warning("No text models found. Install with:")
                st.code("ollama pull llama2:7b")

ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv", "webm", "avi"]
uploaded_files = st.file_uploader(
    "Upload file(s)",
    type=ALLOWED_TYPES,
    accept_multiple_files=True,
)

max_images = st.slider("Max images/pages per file (demo)", min_value=1, max_value=20, value=5)
video_max_frames = st.slider("Max video frames to extract (per video)", min_value=2, max_value=20, value=8)

v_opts = models.get("vision") or [DEFAULT_VISION_MODEL]
t_opts = models.get("text") or [DEFAULT_TEXT_MODEL]
col1, col2 = st.columns(2)
with col1:
    vision_model = st.selectbox("Vision Model", options=v_opts, index=0)
with col2:
    text_model = st.selectbox("Text Synthesis Model", options=t_opts, index=0)


def _save_upload(uploaded_file, index: int) -> Path:
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe = f"{index:03d}_{uploaded_file.name}"
    dest = uploads_dir / safe
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _normalize_uploads() -> list:
    if uploaded_files is None:
        return []
    return list(uploaded_files)


def _prior_qa_block(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    lines: list[str] = []
    for m in msgs:
        label = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{label}: {m.get('content', '')}")
    return "\n\n".join(lines)


files_list = _normalize_uploads()

if files_list:
    st.caption(f"{len(files_list)} file(s) selected.")
    process = st.button("Process & summarize", type="primary")
else:
    process = False


if process and files_list:
    st.session_state.pop("chat_messages", None)
    sources_for_llm: list[dict] = []
    batch_id = int(time.time())
    base_processed = Path("data/processed") / f"batch_{batch_id}"

    with st.spinner("Processing uploads..."):
        for idx, uf in enumerate(files_list):
            pdf_or_media_path = _save_upload(uf, idx)
            stem = Path(uf.name).stem
            run_dir = base_processed / f"{idx:03d}_{stem}"

            try:
                image_paths = media_to_images(
                    pdf_or_media_path,
                    run_dir,
                    video_max_frames=int(video_max_frames),
                )
            except Exception as e:
                msg = str(e)
                st.error(f"Failed on **{uf.name}**: {msg}")
                if "ffmpeg" in msg or "ffprobe" in msg:
                    st.info(
                        "Video support needs either:\n"
                        "1) system `ffmpeg`/`ffprobe` (recommended), or\n"
                        "2) install Python packages `imageio` + `imageio-ffmpeg`.\n\n"
                        "If you're on macOS, `brew install ffmpeg` usually fixes it."
                    )
                st.stop()

            image_paths = image_paths[: int(max_images)]
            is_video = pdf_or_media_path.suffix.lower() in VIDEO_EXTS

            segments: list[tuple[str, str]] = []
            for p in image_paths:
                with st.spinner(f"[{uf.name}] Summarizing {p.name} with {vision_model}..."):
                    summary = visual_summary_from_image(
                        p, source_id=f"{uf.name}/{p.name}", model_name=vision_model
                    )
                segments.append((p.name, summary))

            transcript: str | None = None
            if is_video and segments:
                try:
                    with st.spinner(f"[{uf.name}] Extracting audio..."):
                        audio_path = extract_audio_from_video(pdf_or_media_path, run_dir)
                    with st.spinner(f"[{uf.name}] Transcribing audio (Whisper)..."):
                        transcript = transcribe_audio(audio_path)
                except Exception as e:
                    st.warning(f"[{uf.name}] Audio not used: {e}")

            sources_for_llm.append(
                {
                    "name": uf.name,
                    "segments": segments,
                    "transcript": transcript,
                }
            )

    with st.spinner(f"Synthesizing combined overview with {text_model}..."):
        corpus_summary = synthesize_multi_source_corpus(
            sources_for_llm, model_name=text_model
        )

    full_context = format_multi_source_context(sources_for_llm)

    st.session_state["kb_sources"] = sources_for_llm
    st.session_state["kb_summary"] = corpus_summary
    st.session_state["kb_context"] = full_context
    st.session_state["kb_text_model"] = text_model
    st.session_state["chat_messages"] = []


# --- Display persisted results (survives reruns after Process) ---
if st.session_state.get("kb_summary"):
    st.subheader("Combined summary (all uploads)")
    st.write(st.session_state["kb_summary"])

    sources = st.session_state.get("kb_sources") or []
    if sources:
        st.subheader("Per-segment detail")
        for src in sources:
            name = src["name"]
            for label, summary in src.get("segments") or []:
                with st.expander(f"{name} — {label}"):
                    st.caption(f"Source file: {name}")
                    st.write(summary)
            tr = src.get("transcript")
            if tr and str(tr).strip():
                with st.expander(f"{name} — audio transcript"):
                    st.text(tr[:8000] + ("..." if len(tr) > 8000 else ""))

    st.divider()
    st.subheader("Ask anything about this session")
    st.caption(
        "Uses the combined summary plus full segment/transcript context from your uploads."
    )

    for m in st.session_state.get("chat_messages") or []:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    followup = st.text_area(
        "Your prompt or question",
        placeholder="e.g. List action items, compare the two PDFs, or explain the main argument…",
        height=120,
        key="followup_prompt",
    )

    q_model = st.session_state.get("kb_text_model") or text_model
    if st.button("Generate answer", type="primary"):
        if not (followup or "").strip():
            st.warning("Enter a prompt or question first.")
        else:
            prior_msgs = list(st.session_state.get("chat_messages") or [])
            prior_ex = _prior_qa_block(prior_msgs)
            with st.spinner(f"Generating with {q_model}..."):
                answer = answer_prompt_over_corpus(
                    followup,
                    corpus_summary=st.session_state["kb_summary"],
                    full_context=st.session_state["kb_context"],
                    model_name=q_model,
                    prior_exchanges=prior_ex if prior_ex.strip() else None,
                )
            msgs = st.session_state.setdefault("chat_messages", [])
            msgs.append({"role": "user", "content": followup.strip()})
            msgs.append({"role": "assistant", "content": answer})
