import copy
import time
from pathlib import Path

import streamlit as st

from ingest_logic import VIDEO_EXTS, extract_audio_from_video, media_to_images
from audio_engine import transcribe_audio
# ── Local Ollama engine — no API key needed ───────────────────────────────────
from vision_engine_local import (
    visual_summary_from_image_local   as _vision_fn,
    video_summary_from_frame_summaries_local as _synth_fn,
    check_ollama_models,
    DEFAULT_VISION_MODEL,
    DEFAULT_TEXT_MODEL,
    format_multi_source_context,
    synthesize_multi_source_corpus,
    answer_prompt_over_corpus,
)

st.set_page_config(
    page_title="SecondBrainAI Vision mRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — Neumorphic design
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:          #e8edf2;
  --sd:          #c3c9d4;
  --sl:          #ffffff;
  --accent:      #6c63ff;
  --accent-g:    linear-gradient(135deg,#9b8fff 0%,#6c63ff 100%);
  --accent-soft: rgba(108,99,255,0.12);
  --danger:      #f43f5e;
  --success:     #10b981;
  --text1:       #1e2340;
  --text2:       #4a5060;
  --text3:       #6b7280;
  --display:     1.75rem;   /* hero title */
  --body:        0.9rem;    /* default UI copy */
  --body-sm:     0.85rem;   /* captions, compact labels */
  --ui-xs:       0.8rem;    /* chips, meta */
  --title:       1.1rem;    /* panel / sidebar section titles */
  --r-card:      22px;
  --r-inner:     14px;
  --r-btn:       12px;
  --tr:          all 0.22s ease;
}

/* Inter on shells only — do NOT use `[class*="css"]` here: it matches Streamlit Emotion
   nodes and overrides Material Symbols, so expander chevrons render as "_arrow_right" text. */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
  font-family: 'Inter', sans-serif;
}
.stApp            { background: var(--bg) !important; color: var(--text1) !important; }

/* Global: Streamlit captions — same scale as app subtitle (body + text2) */
.block-container [data-testid="stCaptionContainer"],
.block-container .stCaption,
.block-container [data-testid="stCaptionContainer"] p,
.block-container [data-testid="stCaptionContainer"] span {
  color: var(--text2) !important;
  font-size: var(--body) !important;
  font-weight: 500 !important;
  line-height: 1.5 !important;
}
.block-container [data-testid="stMarkdownContainer"] p {
  color: var(--text1);
  font-size: var(--body);
  line-height: 1.5;
}
.block-container [data-testid="stTextInput"] input {
  font-family: 'Inter', sans-serif !important;
  font-size: var(--body) !important;
  color: var(--text1) !important;
}
/* Prompt box — same neumorphic light field as selectboxes (fixes dark bg / low contrast / red focus) */
[data-testid="stTextArea"] textarea,
.block-container [data-testid="stTextArea"] textarea {
  font-family: 'Inter', sans-serif !important;
  font-size: var(--body) !important;
  color: var(--text1) !important;
  background: var(--bg) !important;
  caret-color: var(--accent) !important;
  border: none !important;
  border-radius: var(--r-inner) !important;
  box-shadow: inset 4px 4px 9px var(--sd), inset -4px -4px 9px var(--sl) !important;
  padding: 0.75rem 1rem !important;
  line-height: 1.5 !important;
  resize: vertical !important;
}
[data-testid="stTextArea"] textarea::placeholder {
  color: var(--text3) !important;
  opacity: 1 !important;
}
[data-testid="stTextArea"] textarea:focus {
  outline: none !important;
  box-shadow: inset 4px 4px 9px var(--sd), inset -4px -4px 9px var(--sl),
    0 0 0 2px rgba(108, 99, 255, 0.28) !important;
}
[data-testid="stTextArea"] [data-baseweb="base-input"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stTextArea"]:focus-within [data-baseweb="base-input"] {
  border-color: transparent !important;
}
.block-container [data-testid="stWidgetLabel"] {
  font-family: 'Inter', sans-serif !important;
  font-size: var(--body-sm) !important;
  font-weight: 600 !important;
  color: var(--text2) !important;
}
/* Help (?) next to labels — Streamlit/BaseWeb default is too light on our cards */
[data-testid="stWidgetLabel"] button,
[data-testid="stWidgetLabel"] [role="button"] {
  color: var(--text2) !important;
  opacity: 1 !important;
}
[data-testid="stWidgetLabel"] button:hover,
[data-testid="stWidgetLabel"] [role="button"]:hover {
  color: var(--accent) !important;
}
[data-testid="stWidgetLabel"] svg,
[data-testid="stWidgetLabel"] svg path {
  fill: currentColor !important;
  stroke: currentColor !important;
  opacity: 1 !important;
}
/* Expander title: size/weight only — chevron uses Material Symbols (don’t force Inter on summary spans) */
[data-testid="stExpander"] summary {
  font-size: var(--body-sm) !important;
  font-weight: 600 !important;
  color: var(--text1) !important;
}
[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
  font-family: 'Inter', sans-serif !important;
  font-size: var(--body-sm) !important;
  font-weight: 600 !important;
  color: var(--text1) !important;
}
[data-testid="stChatMessage"] {
  font-family: 'Inter', sans-serif !important;
  font-size: var(--body) !important;
  color: var(--text1) !important;
}
[data-testid="element-container"] [data-testid="stAlert"] {
  font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container  { padding: 2rem 3rem 4rem !important; max-width: 1180px !important; }

/* ── Processing loader (no grey-out — sliders stay visually enabled) ─────── */
.proc-loader-wrap {
  margin: 0 0 1rem 0;
  padding: 0.85rem 1.15rem;
  border-radius: var(--r-inner);
  background: var(--bg);
  box-shadow: 5px 5px 12px var(--sd), -4px -4px 10px var(--sl);
}
.proc-loader-row {
  display: flex;
  align-items: center;
  gap: 0.85rem;
}
.proc-loader-spin {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(108,99,255,0.22);
  border-top-color: #6c63ff;
  border-radius: 50%;
  animation: sb-spin 0.7s linear infinite;
  flex-shrink: 0;
}
@keyframes sb-spin { to { transform: rotate(360deg); } }
.proc-loader-text {
  font-size: var(--body);
  font-weight: 600;
  color: var(--text1);
  line-height: 1.35;
}

/* ── Column cards ──────────────────────────────────────────────────────── */
[data-testid="stColumn"] {
  background: var(--bg) !important;
  border-radius: var(--r-card) !important;
  padding: 2rem 1.8rem !important;
  box-shadow: 10px 10px 22px var(--sd), -10px -10px 22px var(--sl) !important;
}
[data-testid="stColumn"] [data-testid="stColumn"] {
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0.1rem 0.15rem !important;
  box-shadow: none !important;
}

/* ── Top bar ───────────────────────────────────────────────────────────── */
.top-bar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem; }
.top-bar h1 { font-size:var(--display); font-weight:800; margin:0; color:var(--text1); letter-spacing:-0.5px; }
.top-bar h1 span { color:var(--accent); }
.status-badge {
  display:inline-flex; align-items:center; gap:8px;
  background:var(--bg); border-radius:999px; padding:9px 20px;
  font-size:var(--body-sm); font-weight:600; color:var(--text1);
  box-shadow:5px 5px 10px var(--sd),-5px -5px 10px var(--sl);
}
.dot-pulse { width:9px;height:9px;border-radius:50%;background:var(--success);display:inline-block;
  animation:dp 2s ease-in-out infinite; }
@keyframes dp{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.5)}50%{box-shadow:0 0 0 6px rgba(16,185,129,0)}}
.status-ok  { color:var(--success); font-weight:700; }
.status-err { color:var(--danger);  font-weight:700; }
.dot-err    { width:9px;height:9px;border-radius:50%;background:var(--danger);display:inline-block; }

/* ── Ollama model info chip ────────────────────────────────────────────── */
.model-chip {
  display:inline-flex; align-items:center; gap:6px;
  background:var(--bg); border-radius:999px; padding:5px 14px;
  font-size:var(--ui-xs); font-weight:600; color:var(--accent);
  box-shadow:3px 3px 7px var(--sd),-3px -3px 7px var(--sl);
  margin-right:6px; margin-bottom:4px;
}

/* ── Left column: real Streamlit dropzone (drag & drop works on this element) ─ */
.upload-hint {
  color:var(--text1) !important; font-size:var(--body) !important; font-weight:500 !important;
  line-height:1.5 !important; margin:0 0 0.6rem 0 !important; padding:0 0.1rem;
}
.formats-box {
  text-align:center;font-size:var(--body-sm);color:var(--text2);line-height:1.55;
  margin-top:0.6rem;padding:0.4rem 0.5rem 1.1rem;word-wrap:break-word;max-width:100%;
}
.formats-box strong { color:var(--accent);font-weight:700; }

/* ── File uploader: single column, no horizontal “200MB•types•…” line through the box ─ */
.stFileUploader [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stFileUploader"] > label {
  display:none !important;
}

/* Stack widget vertically; hide redundant long Streamlit type/size paragraph (we use .formats-box) */
[data-testid="stFileUploader"] {
  width:100% !important;
  display:flex !important;
  flex-direction:column !important;
  align-items:stretch !important;
  gap:0.5rem !important;
}
[data-testid="stFileUploader"] > div {
  max-width:100% !important;
}

[data-testid="stFileUploaderDropzone"] {
  border-radius:var(--r-inner) !important;
  box-shadow:inset 5px 5px 11px var(--sd),inset -5px -5px 11px var(--sl) !important;
  border:2px dashed rgba(108,99,255,.38) !important;
  background:var(--bg) !important;
  min-height:158px !important;
  max-width:100% !important;
  box-sizing:border-box !important;
  display:flex !important;
  flex-direction:column !important;
  /* stretch so Browse / “Select files” row is full dropzone width (center was shrink-wrapping the button) */
  align-items:stretch !important;
  justify-content:flex-start !important;
  gap:0.65rem !important;
  padding:1rem 0.85rem !important;
  margin-bottom:0 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  display:flex !important;
  flex-direction:column !important;
  align-items:center !important;
  text-align:center !important;
  gap:0.35rem !important;
  padding:0 !important;
  max-width:100% !important;
  flex-shrink:0 !important;
  align-self:center !important;
  width:100% !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {
  white-space:normal !important;
  word-break:break-word !important;
  overflow-wrap:anywhere !important;
  max-width:100% !important;
  color:var(--text1) !important;
  font-size:var(--body-sm) !important;
  font-weight:600 !important;
  line-height:1.35 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small {
  color:var(--text2) !important;
  font-size:var(--ui-xs) !important;
  font-weight:500 !important;
}
/* Hide the native uploaded-file pill entirely */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderFileData"],
div[class*="uploadedFile"]             { display:none !important; }

/* Wrapper around the browse button: full dropzone width (Streamlit nests button in div rows) */
[data-testid="stFileUploaderDropzone"] div:has(> button),
[data-testid="stFileUploaderDropzone"] div:has(button) {
  width:100% !important;
  max-width:100% !important;
  align-self:stretch !important;
  box-sizing:border-box !important;
}
[data-testid="stFileUploader"] button {
  position:relative !important;
  display:flex !important; align-items:center !important; justify-content:center !important;
  width:100% !important; max-width:100% !important;
  min-width:12rem !important;
  box-sizing:border-box !important;
  flex-shrink:0 !important;
  align-self:stretch !important;
  background:var(--accent-g) !important;
  color:transparent !important; border:none !important;
  border-radius:var(--r-btn) !important; padding:0.68rem 1.25rem !important;
  min-height:2.75rem !important;
  font-size:var(--body) !important; font-weight:700 !important;
  box-shadow:6px 6px 14px rgba(108,99,255,.35),-3px -3px 10px rgba(255,255,255,.6) !important;
  transition:var(--tr) !important;
}
[data-testid="stFileUploader"] button:hover {
  box-shadow:4px 4px 10px rgba(108,99,255,.45),-2px -2px 8px rgba(255,255,255,.5) !important;
  transform:translateY(-1px) !important;
}
[data-testid="stFileUploader"] button:active {
  box-shadow:inset 3px 3px 7px rgba(90,81,238,.4),inset -2px -2px 5px rgba(160,150,255,.3) !important;
  transform:translateY(0) !important;
}
/* One visible label only: hide icon+text children; never clip file <input> if present inside */
[data-testid="stFileUploader"] button *:not(input) {
  position:absolute !important;
  width:1px !important; height:1px !important;
  padding:0 !important; margin:-1px !important;
  overflow:hidden !important; clip:rect(0,0,0,0) !important;
  white-space:nowrap !important; border:0 !important;
}
[data-testid="stFileUploader"] button::after {
  content:"Select files";
  position:absolute !important;
  left:50% !important; top:50% !important;
  transform:translate(-50%,-50%) !important;
  width:max-content !important; max-width:calc(100% - 1rem) !important; height:auto !important;
  margin:0 !important; padding:0 !important;
  overflow:visible !important; clip:auto !important;
  white-space:nowrap !important;
  font-size:var(--body) !important; font-weight:700 !important; color:#fff !important;
  pointer-events:none !important;
}

/* "Upload different file" button (secondary, shown after upload) */
.change-file-btn button[kind="secondary"] {
  width:100% !important; background:var(--bg) !important;
  border:none !important; border-radius:var(--r-btn) !important;
  color:var(--accent) !important; font-size:var(--body) !important; font-weight:700 !important;
  padding:0.62rem 1rem !important;
  box-shadow:5px 5px 10px var(--sd),-5px -5px 10px var(--sl) !important;
  transition:var(--tr) !important;
}
.change-file-btn button[kind="secondary"]:hover {
  box-shadow:3px 3px 7px var(--sd),-3px -3px 7px var(--sl) !important;
  background:var(--accent-soft) !important;
}

/* ── Right panel elements ───────────────────────────────────────────────── */
.panel-title, .page-title {
  font-size: var(--title) !important;
  font-weight: 800 !important;
  color: var(--text1) !important;
  margin: 0 0 0.75rem 0;
  letter-spacing: -0.2px;
  line-height: 1.3;
}
.app-subtitle {
  color: var(--text2) !important;
  font-size: var(--body) !important;
  line-height: 1.5 !important;
  margin-bottom: 1.5rem !important;
  margin-top: -0.9rem !important;
}
.no-file-inset {
  border-radius:var(--r-inner);
  box-shadow:inset 4px 4px 9px var(--sd),inset -4px -4px 9px var(--sl);
  padding:1.1rem 1.2rem; display:flex; align-items:center; gap:10px;
  color:var(--text2); font-size:var(--body-sm); font-weight:500; font-style:normal; margin-bottom:0.8rem;
}
.queue-meta {
  font-size: var(--body-sm) !important;
  font-weight: 600 !important;
  color: var(--text2) !important;
  margin: 0 0 0.5rem 0 !important;
  line-height: 1.45 !important;
}
.nf-icon {
  width:32px;height:32px;border-radius:8px;background:var(--bg);
  box-shadow:3px 3px 6px var(--sd),-3px -3px 6px var(--sl);
  display:inline-flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;
}
.file-card {
  border-radius:var(--r-inner);
  box-shadow:inset 4px 4px 9px var(--sd),inset -4px -4px 9px var(--sl);
  padding:0.9rem 1.1rem; display:flex; align-items:center; gap:12px; margin-bottom:0.4rem;
}
.fc-icon {
  width:38px;height:38px;border-radius:10px;display:inline-flex;
  align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;
  box-shadow:4px 4px 8px var(--sd),-4px -4px 8px var(--sl);
}
.fc-name { font-weight:700;font-size:var(--body-sm);color:var(--text1);
           white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px; }
.fc-size { font-size:var(--ui-xs);color:var(--text2);margin-top:2px;font-weight:500; }
.fc-check {
  margin-left:auto;flex-shrink:0;width:28px;height:28px;border-radius:50%;background:var(--bg);
  box-shadow:3px 3px 6px var(--sd),-3px -3px 6px var(--sl);
  display:inline-flex;align-items:center;justify-content:center;
  color:var(--success);font-size:var(--body-sm);font-weight:700;
}

/* Remove button */
[data-testid="stButton"] button[kind="secondary"].rm-btn { font-size:var(--ui-xs) !important; }

/* ── Section divider ───────────────────────────────────────────────────── */
.sec-divider { height:1px;background:linear-gradient(90deg,transparent,var(--sd),transparent);
               margin:0.9rem 0; }
.sec-label { font-size:var(--ui-xs);font-weight:700;color:var(--text2);
             text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.5rem; }

/* ── Slider sections ───────────────────────────────────────────────────── */
.slider-section {
  border-radius:var(--r-inner);
  box-shadow:inset 3px 3px 7px var(--sd),inset -3px -3px 7px var(--sl);
  padding:0.85rem 1.1rem 0.55rem; margin-bottom:0.7rem;
}
.sl-label { font-size:var(--body-sm);font-weight:600;color:var(--text2); }
.sl-badge {
  background:var(--bg);box-shadow:3px 3px 6px var(--sd),-3px -3px 6px var(--sl);
  border-radius:20px;padding:3px 12px;font-size:var(--ui-xs);font-weight:700;color:var(--accent);
}
[data-testid="stSlider"] > div { padding:0 !important; }
[data-testid="stSlider"] [role="slider"] {
  background:var(--accent-g) !important;
  box-shadow:0 0 0 3px rgba(108,99,255,.2) !important;
  transition:box-shadow .2s ease !important;
}
[data-testid="stSlider"] [role="slider"]:hover { box-shadow:0 0 0 7px rgba(108,99,255,.15) !important; }

/* ── Model selectbox ───────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  border-radius:var(--r-inner) !important;
  border:none !important;
  background:var(--bg) !important;
  box-shadow:inset 3px 3px 7px var(--sd),inset -3px -3px 7px var(--sl) !important;
  font-size:var(--body-sm) !important; color:var(--text1) !important;
}
[data-testid="stSelectbox"] label { font-size:var(--body-sm) !important; font-weight:600 !important;
                                    color:var(--text2) !important; }
/* Selectbox chevron / icons (open indicator) — same contrast as labels */
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stSelectbox"] [data-baseweb="select"] svg path {
  fill: var(--text2) !important;
  color: var(--text2) !important;
  opacity: 1 !important;
}

/* ── Stage indicator ───────────────────────────────────────────────────── */
.stage-chip {
  display:inline-flex;align-items:center;gap:8px;border-radius:999px;padding:8px 16px;
  box-shadow:inset 3px 3px 7px var(--sd),inset -3px -3px 7px var(--sl);
  font-size:var(--body-sm);font-weight:600;color:var(--accent);margin-bottom:0.6rem;width:100%;
}
.stage-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);flex-shrink:0;
  animation:sd 1.1s ease-in-out infinite; }
@keyframes sd{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.7)}}

/* Progress bar */
[data-testid="stProgress"] > div {
  border-radius:999px !important; height:5px !important;
  background:var(--bg) !important;
  box-shadow:inset 2px 2px 4px var(--sd),inset -2px -2px 4px var(--sl) !important;
}
[data-testid="stProgress"] > div > div { background:var(--accent-g) !important; border-radius:999px !important; }

/* ── Action buttons ────────────────────────────────────────────────────── */
[data-testid="stButton"] button[kind="primary"] {
  background:var(--accent-g) !important; border:none !important;
  border-radius:var(--r-btn) !important; color:#fff !important;
  font-weight:700 !important; font-size:var(--body) !important; padding:0.68rem 0.5rem !important;
  box-shadow:7px 7px 15px rgba(108,99,255,.35),-3px -3px 10px rgba(255,255,255,.55) !important;
  transition:var(--tr) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover:not(:disabled) {
  box-shadow:5px 5px 12px rgba(108,99,255,.45),-2px -2px 8px rgba(255,255,255,.5) !important;
  transform:translateY(-1px) !important;
}
[data-testid="stButton"] button[kind="primary"]:active:not(:disabled) {
  box-shadow:inset 4px 4px 9px rgba(90,81,238,.45),inset -2px -2px 6px rgba(160,150,255,.3) !important;
  transform:translateY(0) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
  background:var(--bg) !important; border:none !important;
  border-radius:var(--r-btn) !important; color:var(--danger) !important;
  font-weight:700 !important; font-size:var(--body) !important; padding:0.68rem 0.5rem !important;
  box-shadow:7px 7px 15px var(--sd),-7px -7px 15px var(--sl) !important;
  transition:var(--tr) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover:not(:disabled) {
  background:rgba(244,63,94,.06) !important;
  box-shadow:5px 5px 10px var(--sd),-5px -5px 10px var(--sl) !important;
}
[data-testid="stButton"] button[kind="secondary"]:active:not(:disabled) {
  box-shadow:inset 4px 4px 8px var(--sd),inset -4px -4px 8px var(--sl) !important;
}
[data-testid="stButton"] button:disabled { opacity:.38 !important; cursor:not-allowed !important; }

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border-radius:var(--r-inner) !important; border:none !important;
  box-shadow:5px 5px 10px var(--sd),-5px -5px 10px var(--sl) !important;
  margin-bottom:0.6rem !important; background:var(--bg) !important;
}
[data-testid="stAlert"] {
  border-radius:var(--r-inner) !important; border:none !important;
  box-shadow:5px 5px 10px var(--sd),-5px -5px 10px var(--sl) !important;
  color: var(--text1) !important;
}
/* Only text nodes — avoid styling every inner layout `div` (looks like a duplicate ghost bar) */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] li,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span {
  color: var(--text1) !important;
  font-size: var(--body) !important;
  line-height: 1.5 !important;
}
.results-header { font-size:var(--title);font-weight:800;color:var(--text1);margin:1.8rem 0 0.8rem; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); border-radius:99px; }
::-webkit-scrollbar-thumb { background:var(--sd); border-radius:99px; }

/* ── Sidebar (force readable text — Streamlit/BaseWeb often lightens labels) ─ */
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] [data-baseweb="radio"] label span,
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label span {
  color: var(--text1) !important;
}
[data-testid="stSidebar"] {
  background: var(--bg) !important;
  box-shadow: 4px 0 14px rgba(195,201,212,0.45) !important;
  font-size: var(--body) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] p {
  color: var(--text1) !important;
}
.sidebar-title {
  font-size: var(--title) !important;
  font-weight: 800 !important;
  color: var(--text1) !important;
  margin: 0 0 0.75rem 0 !important;
  letter-spacing: -0.2px;
  line-height: 1.3;
}
/* Radio row labels (option text) */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
  font-size: var(--body) !important;
  font-weight: 600 !important;
  color: var(--text1) !important;
}
[data-testid="stSidebar"] div[data-baseweb="radio"] label {
  font-size: var(--body) !important;
  font-weight: 600 !important;
  color: var(--text1) !important;
}
/* Radio: use app accent (Streamlit default primary is red #FF4B4B) */
[data-testid="stSidebar"] input[type="radio"] {
  accent-color: var(--accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] svg circle[fill="#FF4B4B"],
[data-testid="stSidebar"] [data-baseweb="radio"] svg circle[fill="#ff4b4b"],
[data-testid="stSidebar"] .stRadio svg circle[fill="#FF4B4B"],
[data-testid="stSidebar"] .stRadio svg circle[fill="#ff4b4b"] {
  fill: var(--accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[class*="Circle"] {
  border-color: var(--text3) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] input:checked ~ div {
  border-color: var(--accent) !important;
}
/* Sidebar captions / helper — same family & scale as main captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] span,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] .stCaption span {
  color: var(--text2) !important;
  font-size: var(--body) !important;
  font-weight: 500 !important;
  line-height: 1.5 !important;
  margin-top: 0.25rem !important;
}
/* Primary buttons in sidebar (if any) */
[data-testid="stSidebar"] button[kind="primary"] {
  font-size: var(--body) !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_TYPES = ["pdf", "png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv", "webm", "avi", "mpeg"]

# ── Session state ─────────────────────────────────────────────────────────────
_defaults: dict = {
    "uploaded_files":   None,  # list[UploadedFile] when user has chosen file(s)
    "uploader_key":     0,
    "processing":       False,
    "should_terminate":  False,
    "stage":            "idle",
    "image_paths":      [],
    "process_idx":      0,
    "frame_summaries":  [],
    "frame_labels":     [],
    "is_video":         False,
    "media_path":       None,
    "run_dir":          None,
    "transcript":       None,
    "video_summary":    None,
    "error":            None,
    "max_images_val":   5,
    "video_frames_val": 8,
    "vision_model":     DEFAULT_VISION_MODEL,
    "text_model":       DEFAULT_TEXT_MODEL,
    "batch_queue":      None,  # list[dict] while / after batch run
    "batch_file_idx":   0,
    "batch_sources":    [],   # accumulated per-file segments for synthesis
    "corpus_summary":   None,
    "corpus_context":   None,
    "chat_messages":    [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = copy.deepcopy(_v)

ss = st.session_state
# Migrate older sessions that used ``uploaded_file`` (single) only.
if "uploaded_file" in ss:
    _legacy = ss.pop("uploaded_file", None)
    if _legacy is not None and not ss.get("uploaded_files"):
        ss["uploaded_files"] = [_legacy]

# ── Ollama status (cached per session) ───────────────────────────────────────
@st.cache_data(ttl=30)
def _get_ollama_models():
    return check_ollama_models()

ollama_info   = _get_ollama_models()
ollama_ok     = not bool(ollama_info.get("error"))
vision_models = ollama_info.get("vision", []) or [DEFAULT_VISION_MODEL]
text_models   = ollama_info.get("text",   []) or [DEFAULT_TEXT_MODEL]
all_models    = ollama_info.get("all",    [])


def _save_upload(f, index: int = 0) -> Path:
    d = Path("data/uploads")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{index:03d}_{f.name}"
    p.write_bytes(f.getbuffer())
    return p


def _append_current_file_sources() -> None:
    """Append current file's frame summaries (and video narrative/transcript) to batch_sources."""
    if not ss.batch_queue:
        return
    src_name = ss.batch_queue[ss.batch_file_idx]["name"]
    segments: list[tuple[str, str]] = [
        (Path(ps).name, s) for ps, s in zip(ss.image_paths, ss.frame_summaries)
    ]
    if ss.is_video and ss.video_summary:
        segments.append(("Video narrative (synthesized)", ss.video_summary))
    tr = ss.transcript if ss.is_video else None
    ss.batch_sources.append({"name": src_name, "segments": segments, "transcript": tr})


def _advance_batch_or_finalize() -> None:
    """Move to next file in batch or enter corpus synthesis."""
    if not ss.batch_queue:
        ss.stage = "batch_finalize"
        return
    if ss.batch_file_idx + 1 < len(ss.batch_queue):
        ss.batch_file_idx += 1
        nxt = ss.batch_queue[ss.batch_file_idx]
        ss.media_path = nxt["media_path"]
        ss.run_dir = nxt["run_dir"]
        Path(ss.run_dir).mkdir(parents=True, exist_ok=True)
        ss.image_paths = []
        ss.frame_summaries = []
        ss.frame_labels = []
        ss.process_idx = 0
        ss.is_video = nxt["is_video"]
        ss.transcript = None
        ss.video_summary = None
        ss.stage = "converting"
    else:
        ss.stage = "batch_finalize"


def _prior_qa_block(msgs: list[dict]) -> str:
    if not msgs:
        return ""
    lines: list[str] = []
    for m in msgs:
        label = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{label}: {m.get('content', '')}")
    return "\n\n".join(lines)


def _qa_corpus_summary_text() -> str:
    if ss.corpus_summary and str(ss.corpus_summary).strip():
        return str(ss.corpus_summary).strip()
    return (
        "No combined summary was produced (e.g. processing stopped early). "
        "Answer using only the detailed segment content in the context below."
    )


def _qa_full_context_text() -> str:
    if ss.corpus_context and str(ss.corpus_context).strip():
        return str(ss.corpus_context).strip()
    if ss.batch_sources:
        return format_multi_source_context(ss.batch_sources)
    if ss.frame_summaries and ss.image_paths:
        lines = ["=== PARTIAL CONTEXT (run stopped before batch merge) ==="]
        for ps, s in zip(ss.image_paths, ss.frame_summaries):
            lines.append(f"--- {Path(ps).name} ---\n{s}")
        if ss.transcript:
            lines.append(f"--- AUDIO TRANSCRIPT ---\n{ss.transcript}")
        return "\n\n".join(lines)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  UI  (processing state machine runs at end so Terminate & controls render first)
# ══════════════════════════════════════════════════════════════════════════════

# ── Top bar ───────────────────────────────────────────────────────────────────
if ollama_ok:
    badge_dot, badge_label, badge_val = "dot-pulse", "Ollama", "Connected"
    badge_cls = "status-ok"
else:
    badge_dot, badge_label, badge_val = "dot-err", "Ollama", "Offline"
    badge_cls = "status-err"

st.markdown(f"""
<div class="top-bar">
  <h1>SecondBrainAI&nbsp;<span>Vision&nbsp;mRAG</span></h1>
  <div class="status-badge">
    <span class="{badge_dot}"></span>
    {badge_label}:&nbsp;<span class="{badge_cls}">{badge_val}</span>
  </div>
</div>
<p class="app-subtitle">
  Upload PDFs, images, or videos (one or many) — OCR-free Visual Summaries via local&nbsp;Ollama&nbsp;models.
  &nbsp;<strong style="color:var(--accent); font-weight:700;">No API key required.</strong>
</p>
""", unsafe_allow_html=True)

if not ollama_ok:
    st.warning(
        f"⚠️ Ollama is not running: `{ollama_info.get('error','')}` — "
        "Start it with **`ollama serve`**, then refresh this page."
    )

# ── Sidebar: switch Upload vs Ask ─────────────────────────────────────────────
if "sidebar_nav" not in st.session_state:
    st.session_state.sidebar_nav = "upload"
with st.sidebar:
    st.markdown('<p class="sidebar-title">SecondBrainAI</p>', unsafe_allow_html=True)
    st.radio(
        "Section",
        ["upload", "ask"],
        format_func=lambda x: "📁 Upload & summarize"
        if x == "upload"
        else "💬 Ask (prompt)",
        key="sidebar_nav",
        label_visibility="collapsed",
    )
    st.caption(
        "Upload and run summaries on the first screen; ask questions about the "
        "last completed run on the second."
    )
_nav = st.session_state.sidebar_nav

# ── Full-width processing loader (spinner + stage + progress) ─────────────────
if ss.processing:
    _nq = len(ss.batch_queue) if ss.batch_queue else 1
    _fi = ss.batch_file_idx + 1 if ss.batch_queue else 1
    _proc_labels = {
        "converting": f"File {_fi}/{_nq}: converting media to images…",
        "summarizing": f"File {_fi}/{_nq}: summarizing {ss.process_idx} / {len(ss.image_paths)}…",
        "audio": f"File {_fi}/{_nq}: extracting & transcribing audio…",
        "synthesizing": f"File {_fi}/{_nq}: synthesizing video summary…",
        "batch_finalize": "Synthesizing combined summary…",
    }
    _proc_msg = _proc_labels.get(ss.stage, "Processing…")
    st.markdown(
        f'<div class="proc-loader-wrap"><div class="proc-loader-row">'
        f'<span class="proc-loader-spin"></span>'
        f'<span class="proc-loader-text">{_proc_msg}</span>'
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if ss.stage == "summarizing" and ss.image_paths:
        st.progress(ss.process_idx / max(len(ss.image_paths), 1))
    elif ss.stage == "batch_finalize":
        st.progress(1.0)

start_clicked = False
terminate_clicked = False
files_sel = ss.uploaded_files or []
has_file = len(files_sel) > 0
is_proc = ss.processing

# ── Upload & summarize screen ─────────────────────────────────────────────────
if _nav == "upload":
    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  LEFT PANEL
    #  KEY FIX: conditionally render the file uploader so the native file pill
    #  never appears. When a file is already held in session state we skip the
    #  widget entirely and show "Upload different file" instead.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with col_left:
        st.markdown(
            '<p class="upload-hint">Drag files into the <strong>dashed area</strong> below, '
            "or use <strong>Browse files</strong> to pick one or many PDFs, images, or videos "
            "(Shift/Cmd-click for multiple).</p>",
            unsafe_allow_html=True,
        )
        if not ss.uploaded_files:
            # Native Streamlit dropzone (styled above) — this is the real drag-and-drop target.
            uploaded = st.file_uploader(
                "Choose file(s)",
                type=ALLOWED_TYPES,
                accept_multiple_files=True,
                label_visibility="hidden",
                key=f"up_{ss.uploader_key}",
                disabled=ss.processing,
            )
            if uploaded:
                ss.uploaded_files = list(uploaded)
                st.rerun()
        else:
            # ── Selection held in session: hide uploader until user changes files.
            st.markdown('<div class="change-file-btn">', unsafe_allow_html=True)
            if st.button(
                "🔄  Choose different files",
                use_container_width=True,
                key="change_file",
                disabled=ss.processing,
            ):
                for k, v in _defaults.items():
                    ss[k] = copy.deepcopy(v)
                ss.uploader_key += 1
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="formats-box">
          Supports: PDF, PNG, JPG, JPEG, WEBP<br>
          MP4, MOV, MKV, WEBM, AVI, MPEG
        </div>
        """,
            unsafe_allow_html=True,
        )


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RIGHT PANEL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with col_right:
        st.markdown('<p class="panel-title">Upload &amp; Configure</p>', unsafe_allow_html=True)

        # ── File card(s) ───────────────────────────────────────────────────────────
        if has_file:
            total_mb = sum(len(f.getbuffer()) for f in files_sel) / (1024 * 1024)
            n = len(files_sel)
            st.markdown(
                f'<p class="queue-meta">{n} file{"s" if n != 1 else ""} · {total_mb:.1f} MB total</p>',
                unsafe_allow_html=True,
            )
            for idx, uf in enumerate(files_sel):
                size_mb = len(uf.getbuffer()) / (1024 * 1024)
                ext = Path(uf.name).suffix.lower()
                is_vid = ext in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".m4v"}
                icon = "📄" if ext == ".pdf" else ("🎥" if is_vid else "🖼️")
                icon_bg = "#fde8e8" if ext == ".pdf" else ("#dbeafe" if is_vid else "#d1fae5")
                row_left, row_btn = st.columns([5, 1])
                with row_left:
                    st.markdown(
                        f"""
                <div class="file-card">
                  <div class="fc-icon" style="background:{icon_bg};">{icon}</div>
                  <div style="flex:1;min-width:0;">
                    <div class="fc-name" title="{uf.name}">{uf.name}</div>
                    <div class="fc-size">{size_mb:.1f} MB</div>
                  </div>
                  <div class="fc-check">&#10003;</div>
                </div>
                """,
                        unsafe_allow_html=True,
                    )
                with row_btn:
                    remove_one = st.button(
                        "✕",
                        key=f"rm_one_{ss.uploader_key}_{idx}",
                        disabled=is_proc,
                        help=f"Remove “{uf.name}” from this batch",
                    )
                    if remove_one:
                        new_list = list(ss.uploaded_files or [])
                        if 0 <= idx < len(new_list):
                            new_list.pop(idx)
                        ss.uploaded_files = new_list if new_list else None
                        st.rerun()

            if st.button(
                "Clear all files",
                key="rm_all_files",
                use_container_width=True,
                disabled=is_proc,
            ):
                ss.uploaded_files = None
                ss.uploader_key += 1
                st.rerun()
        else:
            st.markdown("""
            <div class="no-file-inset">
              <div class="nf-icon">📂</div>
              <span>No files selected yet</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)

        # ── Sliders ───────────────────────────────────────────────────────────────
        st.markdown('<div class="slider-section"><p class="sl-label">Max images / pages per file</p>',
                    unsafe_allow_html=True)
        max_images = st.slider(
            "pages", 1, 20, 5, label_visibility="collapsed", key="sl_pages"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin-top:-0.55rem;">'
            f'<span class="sl-badge">Pages: {max_images}</span></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="slider-section"><p class="sl-label">Max video frames to extract</p>',
                    unsafe_allow_html=True)
        video_max_frames = st.slider(
            "frames", 2, 20, 8, label_visibility="collapsed", key="sl_frames"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin-top:-0.55rem;">'
            f'<span class="sl-badge">Frames: {video_max_frames}</span></div></div>',
            unsafe_allow_html=True,
        )

        # ── Ollama model selection ────────────────────────────────────────────────
        # Lists come from ``ollama list`` (see ``check_ollama_models``); ``ollama pull`` adds models.
        # Dropdowns only when there is a real choice; otherwise show the single resolved model.
        st.markdown('<div class="sec-divider"></div><p class="sec-label">Ollama Models</p>',
                    unsafe_allow_html=True)

        _text_opts = text_models if text_models else vision_models
        _pick_vision = len(vision_models) > 1
        _pick_text = len(_text_opts) > 1

        if _pick_vision or _pick_text:
            mc1, mc2 = st.columns(2, gap="small")
            with mc1:
                if _pick_vision:
                    try:
                        _vi = vision_models.index(ss.vision_model)
                    except ValueError:
                        _vi = 0
                    chosen_vision = st.selectbox(
                        "Vision model",
                        options=vision_models,
                        index=_vi,
                        key="sel_vision",
                        help="Used to summarize each image/frame. Add more with `ollama pull` (e.g. llava, qwen2-vl).",
                    )
                    ss.vision_model = chosen_vision
                else:
                    ss.vision_model = vision_models[0]
                    st.markdown(
                        f'<p class="sl-label" style="margin:0 0 0.35rem 0;">Vision model</p>'
                        f'<p class="queue-meta" style="margin:0;">{vision_models[0]}</p>',
                        unsafe_allow_html=True,
                    )
            with mc2:
                if _pick_text:
                    try:
                        _ti = _text_opts.index(ss.text_model)
                    except ValueError:
                        _ti = 0
                    chosen_text = st.selectbox(
                        "Text model",
                        options=_text_opts,
                        index=_ti,
                        key="sel_text",
                        help="Used for video synthesis and combined multi-file summary. Add more with `ollama pull`.",
                    )
                    ss.text_model = chosen_text
                else:
                    ss.text_model = _text_opts[0]
                    st.markdown(
                        f'<p class="sl-label" style="margin:0 0 0.35rem 0;">Text model</p>'
                        f'<p class="queue-meta" style="margin:0;">{_text_opts[0]}</p>',
                        unsafe_allow_html=True,
                    )
        else:
            ss.vision_model = vision_models[0]
            ss.text_model = _text_opts[0]
            st.markdown(
                f'<p class="queue-meta" style="margin:0 0 0.5rem 0;">'
                f"<strong>Vision</strong> {vision_models[0]} &nbsp;·&nbsp; "
                f"<strong>Text</strong> {_text_opts[0]}</p>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Only one local model per role is installed. Run `ollama pull <name>` for more models "
                "to enable switching here."
            )

        if not vision_models or (len(vision_models) == 1 and vision_models[0] == DEFAULT_VISION_MODEL
                                 and DEFAULT_VISION_MODEL not in all_models):
            st.caption("💡 Pull a vision model: `ollama pull llava:7b`")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

        # ── Action buttons ────────────────────────────────────────────────────────
        ba, bb = st.columns(2, gap="small")
        with ba:
            start_label = "⏳  Processing…" if is_proc else "▶  Start Processing"
            start_clicked = st.button(
                start_label,
                type="primary",
                use_container_width=True,
                disabled=(not has_file or is_proc or not ollama_ok),
                key="start",
            )
        with bb:
            terminate_clicked = st.button(
                "⬛  Terminate",
                use_container_width=True,
                disabled=(not is_proc),
                key="terminate",
            )

        if has_file and not is_proc and ollama_ok:
            st.caption(
                "After processing, open **Ask (prompt)** in the sidebar to query your content."
            )

        if has_file and ss.stage in ("done", "terminated"):
            st.info(
                "Summaries appear below. Switch to **Ask (prompt)** in the sidebar for the prompt box."
            )


elif _nav == "ask":
    st.markdown(
        '<p class="page-title">Ask (prompt)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Uses your last finished run: combined summary, per-segment text, and video transcripts."
    )
    _ctx_ask = _qa_full_context_text()
    _ready_ask = ss.stage in ("done", "terminated") and bool(_ctx_ask.strip())
    if not _ready_ask:
        st.info(
            "Go to **Upload & summarize** in the sidebar, add files, and click "
            "**Start Processing**. When the run completes, return here to ask questions."
        )
    else:
        if ss.corpus_summary:
            with st.expander("Quick view: combined summary", expanded=False):
                st.write(ss.corpus_summary)
        for m in ss.chat_messages or []:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        q_prompt = st.text_area(
            "Your question or instruction",
            placeholder="e.g. What are the main themes? Compare these files. List action items…",
            height=120,
            key="app_qa_prompt",
        )
        _ask_text_opts = text_models if text_models else vision_models
        if len(_ask_text_opts) > 1:
            try:
                _ai = _ask_text_opts.index(ss.text_model)
            except ValueError:
                _ai = 0
            chosen_text_ask = st.selectbox(
                "Text model for answers",
                options=_ask_text_opts,
                index=_ai,
                key="sel_text_ask",
                help="Same source as Upload & summarize: local `ollama list`. Pull more to choose here.",
            )
            ss.text_model = chosen_text_ask
        else:
            ss.text_model = _ask_text_opts[0]
            st.markdown(
                f'<p class="sl-label" style="margin:0 0 0.25rem 0;">Text model for answers</p>'
                f'<p class="queue-meta" style="margin:0 0 0.75rem 0;">{_ask_text_opts[0]}</p>',
                unsafe_allow_html=True,
            )
        if st.button("Generate answer", type="primary", key="app_qa_go"):
            if not (q_prompt or "").strip():
                st.warning("Enter a prompt or question first.")
            else:
                prior = _prior_qa_block(list(ss.chat_messages or []))
                with st.spinner(f"Generating with {ss.text_model}…"):
                    ans = answer_prompt_over_corpus(
                        q_prompt,
                        corpus_summary=_qa_corpus_summary_text(),
                        full_context=_ctx_ask,
                        model_name=ss.text_model,
                        prior_exchanges=prior if prior.strip() else None,
                    )
                ss.chat_messages = list(ss.chat_messages or [])
                ss.chat_messages.append({"role": "user", "content": q_prompt.strip()})
                ss.chat_messages.append({"role": "assistant", "content": ans})
                st.rerun()


# ── Errors & notices ──────────────────────────────────────────────────────────
if ss.error:
    st.error(f"Error: {ss.error}")
    if "ffmpeg" in ss.error or "ffprobe" in ss.error:
        st.info("Video needs ffmpeg, or: `pip install imageio imageio-ffmpeg`")

if ss.stage == "terminated":
    st.warning("Processing terminated — partial results shown below (if any).")


# ── Start handler ─────────────────────────────────────────────────────────────
if start_clicked and has_file and not is_proc:
    max_images = int(st.session_state.get("sl_pages", 5))
    video_max_frames = int(st.session_state.get("sl_frames", 8))
    files = list(ss.uploaded_files or [])
    ts = int(time.time())
    ss.batch_queue = []
    for i, f in enumerate(files):
        mp = _save_upload(f, i)
        stem = Path(f.name).stem
        rd = Path("data/processed") / f"run_{ts}_{i:02d}_{stem}"
        rd.mkdir(parents=True, exist_ok=True)
        ext = Path(f.name).suffix.lower()
        ss.batch_queue.append(
            {
                "media_path": str(mp),
                "run_dir": str(rd),
                "name": f.name,
                "is_video": ext in VIDEO_EXTS,
            }
        )
    ss.batch_file_idx = 0
    ss.batch_sources = []
    ss.corpus_summary = None
    ss.corpus_context = None
    first = ss.batch_queue[0]
    ss.media_path = first["media_path"]
    ss.run_dir = first["run_dir"]
    ss.is_video = first["is_video"]
    ss.max_images_val = max_images
    ss.video_frames_val = video_max_frames
    ss.processing = True
    ss.should_terminate = False
    ss.stage = "converting"
    ss.frame_summaries = []
    ss.frame_labels = []
    ss.image_paths = []
    ss.error = None
    ss.video_summary = None
    ss.transcript = None
    ss.chat_messages = []
    st.rerun()

if terminate_clicked and is_proc:
    ss.should_terminate = True


# ── Results (summaries) — Upload & summarize screen only ─────────────────────
_done_like = ss.stage in ("done", "terminated")
_has_segments = bool(ss.batch_sources) or bool(ss.frame_summaries)
if _nav == "upload" and _done_like and _has_segments:
    if ss.corpus_summary:
        st.markdown('<p class="results-header">Combined summary (all files)</p>', unsafe_allow_html=True)
        st.write(ss.corpus_summary)

    if ss.batch_sources:
        st.markdown('<p class="results-header">Per-segment detail</p>', unsafe_allow_html=True)
        for src in ss.batch_sources:
            sname = src["name"]
            for label, summary in src.get("segments") or []:
                with st.expander(f"{sname} — {label}"):
                    st.caption(sname)
                    st.write(summary)
            tr = src.get("transcript")
            if tr and str(tr).strip():
                with st.expander(f"{sname} — audio transcript"):
                    st.text(tr[:5000] + ("…" if len(tr) > 5000 else ""))
    elif ss.frame_summaries:
        if ss.is_video and ss.video_summary:
            st.markdown('<p class="results-header">Video Summary</p>', unsafe_allow_html=True)
            st.write(ss.video_summary)
            st.caption(f"Source: {Path(ss.media_path).name}")
        if ss.transcript:
            with st.expander("View transcript"):
                st.text(ss.transcript[:5000] + ("…" if len(ss.transcript) > 5000 else ""))
        st.markdown('<p class="results-header">Per-frame detail</p>', unsafe_allow_html=True)
        for path_str, summary in zip(ss.image_paths, ss.frame_summaries):
            with st.expander(Path(path_str).name):
                st.caption(path_str)
                st.write(summary)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESSING STATE MACHINE (runs last so the UI above is always rendered first)
# ══════════════════════════════════════════════════════════════════════════════

if ss.stage == "batch_finalize":
    if ss.should_terminate:
        ss.processing = False
        ss.stage = "terminated"
        st.rerun()
    try:
        if ss.batch_sources:
            ss.corpus_summary = synthesize_multi_source_corpus(
                ss.batch_sources, model_name=ss.text_model
            )
            ss.corpus_context = format_multi_source_context(ss.batch_sources)
        else:
            ss.corpus_summary = None
            ss.corpus_context = None
    except Exception as exc:
        ss.error = str(exc)
        ss.corpus_summary = None
        ss.corpus_context = None
    ss.stage = "done"
    ss.processing = False
    st.rerun()

elif ss.stage == "converting":
    if ss.should_terminate:
        ss.stage = "terminated"
        ss.processing = False
        st.rerun()
    try:
        paths = media_to_images(
            Path(ss.media_path), Path(ss.run_dir),
            video_max_frames=ss.video_frames_val,
        )
        paths = paths[: ss.max_images_val]
        ss.image_paths = [str(p) for p in paths]
        ss.is_video = Path(ss.media_path).suffix.lower() in VIDEO_EXTS
        ss.process_idx = 0
        ss.frame_summaries = []
        ss.frame_labels = []
        ss.stage = "summarizing"
    except Exception as exc:
        ss.error = str(exc)
        ss.stage = "idle"
        ss.processing = False
    st.rerun()

elif ss.stage == "summarizing":
    if ss.should_terminate:
        ss.stage = "terminated"
        ss.processing = False
        st.rerun()
    elif ss.process_idx < len(ss.image_paths):
        p = Path(ss.image_paths[ss.process_idx])
        summary = _vision_fn(p, source_id=str(p), model_name=ss.vision_model)
        ss.frame_summaries.append(summary)
        ss.frame_labels.append(p.stem)
        ss.process_idx += 1
        st.rerun()
    else:
        if ss.is_video:
            ss.stage = "audio"
        else:
            if ss.batch_queue:
                _append_current_file_sources()
                _advance_batch_or_finalize()
            else:
                ss.stage = "done"
                ss.processing = False
        st.rerun()

elif ss.stage == "audio":
    if ss.should_terminate:
        ss.stage = "terminated"
        ss.processing = False
        st.rerun()
    try:
        audio_path = extract_audio_from_video(Path(ss.media_path), Path(ss.run_dir))
        ss.transcript = transcribe_audio(audio_path)
    except Exception:
        ss.transcript = None
    ss.stage = "synthesizing"
    st.rerun()

elif ss.stage == "synthesizing":
    if ss.should_terminate:
        ss.stage = "terminated"
        ss.processing = False
        st.rerun()
    ss.video_summary = _synth_fn(
        ss.frame_summaries,
        frame_labels=ss.frame_labels,
        transcript=ss.transcript,
        model_name=ss.text_model,
    )
    if ss.batch_queue:
        _append_current_file_sources()
        _advance_batch_or_finalize()
    else:
        ss.stage = "done"
        ss.processing = False
    st.rerun()
