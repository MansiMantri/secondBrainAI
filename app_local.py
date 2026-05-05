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
    video_summary_from_frame_summaries,
    check_ollama_models,
    DEFAULT_VISION_MODEL,
    DEFAULT_TEXT_MODEL
)


st.set_page_config(page_title="SecondBrainAI Local (Ollama)", layout="centered")
st.title("SecondBrainAI Vision mRAG - Local Version")

st.write("Upload a PDF, image, or video. Uses local Ollama models for OCR-free Visual Summaries.")
st.write("**No API keys required** - runs entirely on your machine.")

# Check Ollama status
with st.expander("🔍 Ollama Status"):
    models = check_ollama_models()
    if models.get('error'):
        st.error(f"Ollama connection failed: {models['error']}")
        st.info("Make sure Ollama is running: `ollama serve`")
    else:
        st.success("Ollama connected successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vision Models")
            if models['vision']:
                for model in models['vision']:
                    st.code(model)
            else:
                st.warning("No vision models found. Install with:")
                st.code("ollama pull llava:7b")

        with col2:
            st.subheader("Text Models")
            if models['text']:
                for model in models['text']:
                    st.code(model)
            else:
                st.warning("No text models found. Install with:")
                st.code("ollama pull llama2:7b")

ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv", "webm", "avi"]
uploaded = st.file_uploader("Upload a file", type=ALLOWED_TYPES)

max_images = st.slider("Max images/pages to summarize (demo)", min_value=1, max_value=20, value=5)
video_max_frames = st.slider("Max video frames to extract", min_value=2, max_value=20, value=8)

# Model selection
col1, col2 = st.columns(2)
with col1:
    vision_model = st.selectbox(
        "Vision Model",
        options=models.get('vision', [DEFAULT_VISION_MODEL]),
        index=0 if models.get('vision') else None
    )

with col2:
    text_model = st.selectbox(
        "Text Synthesis Model",
        options=models.get('text', [DEFAULT_TEXT_MODEL]),
        index=0 if models.get('text') else None
    )


def _save_upload(uploaded_file) -> Path:
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


if uploaded is not None:
    with st.spinner("Converting to images..."):
        t0 = time.time()
        pdf_or_media_path = _save_upload(uploaded)
        run_dir = Path("data/processed") / f"run_{int(time.time())}_{pdf_or_media_path.stem}"

        try:
            image_paths = media_to_images(
                pdf_or_media_path,
                run_dir,
                video_max_frames=int(video_max_frames),
            )
            st.success(f"Converted to {len(image_paths)} image(s) in {time.time() - t0:.1f}s.")
        except Exception as e:
            msg = str(e)
            st.error(f"Failed to convert media to images: {msg}")
            if "ffmpeg" in msg or "ffprobe" in msg:
                st.info(
                    "Video support needs either:\n"
                    "1) system `ffmpeg`/`ffprobe` (recommended), or\n"
                    "2) install Python packages `imageio` + `imageio-ffmpeg`.\n\n"
                    "If you're on macOS, `brew install ffmpeg` usually fixes it."
                )
            st.stop()

    # Keep it bounded for a responsive UI.
    image_paths = image_paths[: int(max_images)]
    is_video = pdf_or_media_path.suffix.lower() in VIDEO_EXTS

    # Per-frame or per-page visual summaries (for RAG and optional detail).
    frame_summaries: list[str] = []
    frame_labels: list[str] = []

    for p in image_paths:
        with st.spinner(f"Summarizing {p.name} with {vision_model}..."):
            summary = visual_summary_from_image(p, source_id=str(p), model_name=vision_model)
        frame_summaries.append(summary)
        frame_labels.append(p.stem)

    if is_video and len(frame_summaries) > 0:
        # Extract and transcribe audio so the summary can include what was said.
        transcript: str | None = None
        try:
            with st.spinner("Extracting audio..."):
                audio_path = extract_audio_from_video(pdf_or_media_path, run_dir)
            with st.spinner("Transcribing audio (Whisper)..."):
                transcript = transcribe_audio(audio_path)
            if transcript:
                with st.expander("View transcript"):
                    st.text(transcript[:5000] + ("..." if len(transcript) > 5000 else ""))
        except Exception as e:
            st.warning(f"Audio was not used (extraction or transcription failed): {e}")

        # One cohesive summary for the entire video (visuals + transcript if available).
        st.subheader("Video Summary (whole video)")
        with st.spinner(f"Synthesizing with {text_model}..."):
            video_summary = video_summary_from_frame_summaries(
                frame_summaries,
                frame_labels=frame_labels,
                transcript=transcript,
                model_name=text_model
            )
        st.write(video_summary)
        st.caption(f"Source: {pdf_or_media_path.name}")
        st.subheader("Per-frame detail (for retrieval)")
        for p, summary in zip(image_paths, frame_summaries):
            with st.expander(p.name):
                st.caption(f"Source: {p}")
                st.write(summary)
    else:
        # PDF or image: show each visual summary as before.
        st.subheader("Visual Summaries")
        for p, summary in zip(image_paths, frame_summaries):
            with st.expander(p.name):
                st.caption(f"Source: {p}")
                st.write(summary)
