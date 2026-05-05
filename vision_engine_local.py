"""
Local Vision Engine for mRAG - Ollama-based alternative to Gemini API.

Uses Ollama with LLaVA or other vision-capable models for local inference.
No API keys required, runs entirely on local hardware.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Optional, Union

import ollama  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]

from model_guidelines import compose_system_prompt


# Prefer newer instruct models when present (`ollama pull llama3`, `qwen2.5`, etc.).
DEFAULT_VISION_MODEL = "llava:13b"
DEFAULT_TEXT_MODEL = "llama3"

PREFERRED_TEXT_MODELS: tuple[str, ...] = (
    "llama3.3",
    "llama3.2",
    "llama3.1",
    "llama3",
    "qwen3",
    "qwen2.5",
    "qwen2",
    "qwen",
    "mistral-small",
    "mistral-nemo",
    "mistral",
    "mixtral",
    "phi4",
    "phi3",
    "gemma3",
    "gemma2",
    "deepseek-r1",
    "deepseek",
    "llama2",
)

PREFERRED_VISION_MODELS: tuple[str, ...] = (
    "llava:34b",
    "llava:13b",
    "llava:7b",
    "llava",
    "bakllava",
    "moondream",
    "minicpm-v",
    "qwen2.5vl",
    "qwen-vl",
    "gemma3",
    "granite3.2-vision",
    "pixtral",
)


def pick_preferred_model(available: list[str], preferences: tuple[str, ...]) -> Optional[str]:
    """Pick the first installed model that matches a preference substring (order = priority)."""
    if not available:
        return None
    for pref in preferences:
        pl = pref.lower()
        for name in available:
            nl = name.lower()
            if pl == nl or pl in nl or nl.startswith(pl + ":") or nl.startswith(pref + "/"):
                return name
    return available[0]


def pick_default_text_model(ollama_info: dict[str, Any]) -> str:
    names = ollama_info.get("text") or []
    if not names:
        return DEFAULT_TEXT_MODEL
    return pick_preferred_model(names, PREFERRED_TEXT_MODELS) or names[0]


def pick_default_vision_model(ollama_info: dict[str, Any]) -> str:
    names = ollama_info.get("vision") or []
    if not names:
        return DEFAULT_VISION_MODEL
    return pick_preferred_model(names, PREFERRED_VISION_MODELS) or names[0]


def _encode_image_to_base64(image_input: Union[str, Path, bytes, Image.Image]) -> str:
    """Convert image to base64 string for Ollama API."""
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(bytes(image_input))).convert("RGB")
    else:
        raise TypeError("image_input must be a path/str, bytes, or PIL.Image.Image.")

    # Convert to RGB and save as JPEG for smaller size
    buffer = io.BytesIO()
    # Higher quality preserves small text on PDF/slide renders for the vision model.
    pil_img.save(buffer, format="JPEG", quality=92)
    buffer.seek(0)
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


def _extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON object from model response text."""
    text = text.strip()

    # Look for JSON-like structure
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def visual_summary_from_image_local(
    image_input: Union[str, Path, bytes, Image.Image],
    *,
    source_id: Optional[str] = None,
    model_name: str = DEFAULT_VISION_MODEL,
    temperature: float = 0.1,
    page_extracted_text: Optional[str] = None,
) -> str:
    """
    Generate visual summary using local Ollama model (LLaVA).

    Returns:
        Visual summary text describing the image content.
    """
    # Encode image to base64
    image_b64 = _encode_image_to_base64(image_input)

    has_embedded = bool(page_extracted_text and page_extracted_text.strip())

    if has_embedded:
        # Text layer already captured — avoid redundant long “summary of the paper” from vision.
        _px = page_extracted_text.strip()
        if len(_px) > 12_000:
            _px = _px[:12_000] + "\n\n[…truncated for model context…]"
        system_prompt = compose_system_prompt(
            "The PDF text for this page is already extracted elsewhere. "
            "Your ONLY task: describe visuals and infographics the plain text does not fully encode — "
            "figures, flowcharts, architecture/stack diagrams, timelines, screenshots, maps, charts, tables, "
            "multi-panel layouts, icons/legend colors when readable. "
            "Prioritize infographics: nodes, arrows, swimlanes, tiers, layers, and how components connect. "
            "Length: about 70–180 words (use the upper range when several figures or a dense diagram appear). "
            "Do NOT paraphrase long narrative paragraphs from the page — the reader has the text extract. "
            "FORBIDDEN phrasing (do not use): ‘The image appears’, ‘This page’, ‘The content’, "
            "‘The document/paper discusses’, ‘The page outlines’, ‘The user workflow is described’, "
            "‘Overall’, ‘In summary’, ‘suggests that’. "
            "Prefer starting with a concrete anchor: e.g. ‘Figure 2: …’, ‘Screenshot: …’, ‘Three-tier diagram: …’. "
            "If the page is nearly all body text with no distinct figures or UI, reply exactly one line: "
            "Mostly body text; no figures or UI beyond the embedded text."
        )
        user_prompt = (
            "Embedded text for this page (do not quote or summarize):\n"
            f"---\n{_px}\n---\n\n"
            "Write the visual/layout supplement only — no meta commentary about what the page is ‘about’."
        )
        if source_id:
            user_prompt += f"\n\nPage: `{source_id}`"
        _num_predict = 400
    else:
        # No text layer (e.g. scan) — full vision description.
        system_prompt = compose_system_prompt(
            "You summarize ONE document page or slide image for retrieval (scanned PDF or image-only). "
            "Use ONLY what is visible in this image. "
            "For slides and PDF pages: quote readable titles, headings, and short labels exactly when legible. "
            "Describe diagrams, charts, tables, and figures in detail: labels, axes, arrows, and relationships. "
            "If text is blurry or too small, say the text is unreadable—do not guess the topic from vague shapes. "
            "Do not infer domains unless those words or unmistakable branding appear in the image. "
            "Do not invent steps, statistics, or bullet lists not clearly shown. "
            "Be thorough: several paragraphs when the page is dense. Only this single image—not a multi-slide story."
        )

        user_prompt = (
            "Summarize ONLY this one page/slide in detail.\n"
            "1) Readable text: headings, key phrases, and notable quotes you can read.\n"
            "2) Visuals: diagrams, tables, charts, screenshots — structure and what they communicate.\n"
            "3) If little or no text is readable, say so plainly and rely on what you can still see.\n"
            "4) Close with what this page contributes to the document, grounded strictly in (1)–(2)."
        )

        if source_id:
            user_prompt += f"\n\nFilename context (not content): `{source_id}`"
        _num_predict = 1400

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': user_prompt,
                    'images': [image_b64]
                }
            ],
            options={
                'temperature': (0.05 if has_embedded else temperature),
                'num_predict': _num_predict,
            }
        )

        summary = response['message']['content'].strip()

        # Try to extract structured info if the model provides it
        json_data = _extract_json_from_text(summary)
        if json_data and 'visual_summary' in json_data:
            return json_data['visual_summary']

        return summary

    except Exception as e:
        raise RuntimeError(f"Local vision model failed: {e}") from e


def detailed_page_summary_from_extract_and_visual(
    extracted_text: str,
    visual_notes: str,
    *,
    source_id: Optional[str] = None,
    model_name: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.18,
) -> str:
    """
    Turn authoritative PDF text plus vision-only notes into one detailed, reader-facing page summary.

    Used so the UI shows polished prose instead of raw ``=== Text extract ===`` dumps.
    """
    et = (extracted_text or "").strip()
    vn = (visual_notes or "").strip()
    if not et and not vn:
        return "No content for this page."

    if len(et) > 16_000:
        et = et[:16_000] + "\n\n[…truncated for model context…]"
    if len(vn) > 6_000:
        vn = vn[:6_000] + "\n\n[…truncated…]"

    nch = len(et)
    if nch < 1_200:
        depth = "roughly 200–500 words"
    elif nch < 6_000:
        depth = "roughly 450–900 words"
    else:
        depth = "roughly 700–1400 words"

    system_prompt = compose_system_prompt(
        "You write a thorough, reader-facing summary of ONE document page for someone who will not open the PDF. "
        "The PDF text extract below is authoritative for facts, terminology, names, numbers, and citations. "
        "Visual notes describe figures, diagrams, charts, screenshots, and layout — weave them in where they "
        "add structure or clarify what the text refers to (e.g. ‘Figure 2 shows …’). "
        f"Target depth: {depth}; scale upward when the extract is long or dense (e.g. methods, studies). "
        "Use multiple paragraphs with logical flow. "
        "Do not use ASCII banner lines or fences made of equals signs (===). "
        "Do not claim you lack the document. "
        "Preserve technical precision; keep module lists, phase names, and procedure steps when present."
    )

    user_prompt = (
        "Produce the detailed page summary.\n\n"
        f"### PDF text extract (authoritative)\n{et}\n\n"
        f"### Visual / figure notes (from the rendered page image)\n{vn or '(No separate visual detail.)'}\n"
    )
    if source_id:
        user_prompt += f"\nPage: `{source_id}`\n"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": 4096,
            },
        )
        out = response["message"]["content"].strip()
        return out if out else et[:4000] + ("\n…" if len(et) > 4000 else "")
    except Exception as e:
        raise RuntimeError(f"Local page-detail model failed: {e}") from e


def video_summary_from_frame_summaries_local(
    frame_summaries: list[str],
    frame_labels: Optional[list[str]] = None,
    *,
    transcript: Optional[str] = None,
    model_name: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.3,
) -> str:
    """
    Synthesize video summary using local Ollama text model.

    Combines frame summaries and transcript into cohesive narrative.
    """
    if not frame_summaries and not (transcript and transcript.strip()):
        return "No content to summarize."

    # Build context from frame summaries
    context_parts = []
    if frame_summaries:
        for i, summary in enumerate(frame_summaries):
            label = (frame_labels[i] if frame_labels and i < len(frame_labels) else f"Frame {i + 1}")
            context_parts.append(f"--- {label} ---\n{summary.strip()}")

    context = "\n\n".join(context_parts)

    # Add transcript if available
    if transcript and transcript.strip():
        context += f"\n\n--- AUDIO TRANSCRIPT ---\n{transcript.strip()}"

    # Create synthesis prompt
    system_prompt = (
        "You are a helpful assistant that creates cohesive video summaries. "
        "Combine visual descriptions and audio content into a single, flowing narrative. "
        "When frame descriptions mention slides, charts, diagrams, or on-screen text, include that material "
        "if it is central to the message (not every incidental graphic)."
    )

    user_prompt = (
        "Based on the following frame descriptions"
        + (" and audio transcript" if transcript else "") +
        ", create ONE cohesive summary of the entire video.\n\n"
        "Guidelines:\n"
        "- Write as a continuous narrative, not a list\n"
        "- Include both visual and audio elements\n"
        "- Retain important on-screen graphics (charts, diagrams, labeled slides) described in the frames\n"
        "- Describe the overall flow and story\n"
        "- Be concise but comprehensive\n"
        "- Use transitions like 'then', 'next', 'finally'\n\n"
        f"Content:\n{context}"
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                'temperature': temperature,
                'num_predict': 1024,  # Allow longer synthesis
            }
        )

        return response['message']['content'].strip()

    except Exception as e:
        raise RuntimeError(f"Local synthesis model failed: {e}") from e


def check_ollama_models() -> dict[str, list[str]]:
    """
    Check which Ollama models are available locally.

    Returns:
        Dict with 'vision' and 'text' model lists.
    """
    try:
        lr = ollama.list()
        raw = lr.get('models', []) if isinstance(lr, dict) else getattr(lr, 'models', [])
        model_names: list[str] = []
        for m in raw:
            if isinstance(m, dict):
                n = m.get('name') or m.get('model')
            else:
                n = getattr(m, 'model', None) or getattr(m, 'name', None)
            if n:
                model_names.append(n)

        _vision_markers = (
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen-vl",
            "llama3.2-vision",
            "gemma3",
            "granite3.2-vision",
            "mistral-small3.1",
            "mistral-small3.2",
            "internvl",
            "phi4",
            "glm-4",
            "pixtral",
            "vision",
            "-vl",
            "vl-",
            "vl:",
            "vl ",
            "4o",
            "4.1",
            "gpt-4",
            "claude-3",
            "gemini",
            "flash",
            "pro-vision",
        )
        vision_models = [
            name for name in model_names if any(m in name.lower() for m in _vision_markers)
        ]
        text_models = [name for name in model_names if name not in vision_models]

        return {
            'vision': vision_models,
            'text': text_models,
            'all': model_names
        }

    except Exception as e:
        return {
            'vision': [],
            'text': [],
            'all': [],
            'error': str(e)
        }


def format_multi_source_context(
    sources: list[dict[str, Any]],
    *,
    include_verbatim: bool = True,
) -> str:
    """
    Build a single text block from processed sources for LLM context.

    Each source dict should include:
      - 'name': str (display filename)
      - 'segments': list[tuple[str, str]] (label, per-page or per-frame summary)
      - 'verbatim_segments': optional list[tuple[str, str]] (label, raw PDF text for that page)
      - 'transcript': optional str (e.g. video audio)

    For corpus **synthesis**, use ``include_verbatim=False`` so the model sees polished per-page
    summaries only (avoids duplicating full raw extracts). For **Q&A**, use
    ``include_verbatim=True`` so answers can ground in exact wording.
    """
    blocks: list[str] = []
    for src in sources:
        name = str(src.get("name", "unknown"))
        blocks.append(f"=== SOURCE: {name} ===")
        for label, summ in src.get("segments") or []:
            blocks.append(f"--- {label} (summary) ---\n{str(summ).strip()}")
        if include_verbatim:
            for label, raw in src.get("verbatim_segments") or []:
                t = str(raw).strip()
                if t:
                    blocks.append(f"--- {label} (verbatim PDF text) ---\n{t}")
        tr = src.get("transcript")
        if tr and str(tr).strip():
            blocks.append(f"--- AUDIO TRANSCRIPT ({name}) ---\n{str(tr).strip()}")
    return "\n\n".join(blocks)


def synthesize_multi_source_corpus(
    sources: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.25,
) -> str:
    """
    One cohesive summary across multiple uploaded documents/media.

    Each item in ``sources`` matches :func:`format_multi_source_context`.
    """
    if not sources:
        return "No content to summarize."

    context = format_multi_source_context(sources, include_verbatim=False)
    if not context.strip():
        return "No content to summarize."

    system_prompt = compose_system_prompt(
        "You write a single polished overview of the material below for a technical reader. "
        "The excerpts are per-page (or per-frame) summaries already distilled from the uploads — "
        "treat them as accurate representations of the source material. "
        "You MUST integrate important non-text content when the excerpts describe it: figures, diagrams, "
        "flowcharts, architecture drawings, charts, maps, screenshots, and infographics. "
        "Name figure numbers or slide titles when given; summarize structure (components, flows, layers, relationships) "
        "that diagrams convey — this is not optional fluff if visuals carry load-bearing information. "
        "Avoid duplicating the SAME sentence-level claim twice; but DO include diagram topology, workflow arrows, "
        "and chart trends even when related topics appear in text, because visuals add structure the prose omits. "
        "Style: clear, professional; several cohesive paragraphs when the material is rich. "
        "Avoid meta filler (‘this document discusses’, ‘the page outlines’, ‘the authors explain’). "
        "State facts directly; do not describe what ‘the text says’ about the topic. "
        "Open with the concrete subject (e.g. system name, paper topic, problem domain) in the first sentence. "
        "No bullet lists unless the source is inherently list-like. "
        "Grounding rule: the user message below contains processed content from the user’s upload(s). "
        "Never claim you lack the document, need an attachment, or only have a third-party ‘description’ of it."
    )

    user_prompt = (
        "Create ONE unified summary from the per-page summaries below (and optional audio transcripts). "
        "This material IS derived from the user’s uploaded files — use it directly.\n"
        "- Merge overlapping narrative across pages/sources.\n"
        "- Explicitly weave in figures/infographics where those summaries describe them.\n"
        "- Target roughly 500–1200 words when multiple pages or sources are present; shorter if the corpus is thin.\n"
        "- Do not introduce claims not supported by the excerpts.\n\n"
        f"{context}"
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={
                'temperature': temperature,
                'num_predict': 3072,
            },
        )
        return response['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Local synthesis model failed (multi-source): {e}") from e


def answer_prompt_over_corpus(
    user_prompt: str,
    *,
    corpus_summary: str,
    full_context: str,
    model_name: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.4,
    prior_exchanges: Optional[str] = None,
) -> str:
    """
    Answer a user question using the summarized corpus and full segment/transcript context.

    Optional ``prior_exchanges`` is formatted earlier Q&A in this session (for follow-up prompts).
    """
    q = (user_prompt or "").strip()
    if not q:
        return "Enter a question or instruction."

    system_prompt = compose_system_prompt(
        "You respond using ONLY the synthesis and detailed context provided below. "
        "If something cannot be supported by that material, say clearly that it is not stated there. "
        "Stay focused on the user's request. "
        "When the context describes figures, diagrams, charts, or screenshots, incorporate those details "
        "if they help answer the question — do not ignore visual-only information in the segments. "
        "Never claim you lack the original file, cannot see the PDF, or only have a ‘description’—the detailed "
        "context IS the processed upload (text extracts and per-page visual notes). If something is missing, "
        "say it is not in the provided excerpts, not that no document was supplied. "
        "If prior Q&A is included, use it only for conversational continuity — facts must still come "
        "from the document context."
    )

    detail = full_context.strip()
    max_detail = 120_000
    if len(detail) > max_detail:
        detail = detail[:max_detail] + "\n\n[... truncated for length ...]"

    prior_block = ""
    if prior_exchanges and prior_exchanges.strip():
        pe = prior_exchanges.strip()
        max_prior = 16_000
        if len(pe) > max_prior:
            pe = pe[:max_prior] + "\n\n[... prior Q&A truncated ...]"
        prior_block = f"=== PRIOR Q&A (this session) ===\n{pe}\n\n"

    user_message = (
        "=== COMBINED SUMMARY (high level) ===\n"
        f"{corpus_summary.strip()}\n\n"
        "=== DETAILED CONTEXT (segments and transcripts) ===\n"
        f"{detail}\n\n"
        f"{prior_block}"
        "=== USER REQUEST ===\n"
        f"{q}"
    )

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
            ],
            options={
                'temperature': temperature,
                'num_predict': 2048,
            },
        )
        return response['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Local answer model failed: {e}") from e


# Backwards compatibility - alias the local functions
visual_summary_from_image = visual_summary_from_image_local
video_summary_from_frame_summaries = video_summary_from_frame_summaries_local
