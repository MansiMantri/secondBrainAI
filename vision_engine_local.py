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


DEFAULT_VISION_MODEL = "llava:7b"  # or "llava:13b" for better quality
DEFAULT_TEXT_MODEL = "llama2:7b"   # for synthesis tasks


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
    pil_img.save(buffer, format="JPEG", quality=85)
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
    temperature: float = 0.2,
) -> str:
    """
    Generate visual summary using local Ollama model (LLaVA).

    Returns:
        Visual summary text describing the image content.
    """
    # Encode image to base64
    image_b64 = _encode_image_to_base64(image_input)

    # Create prompt for vision model (universal guidelines + task-specific role)
    system_prompt = compose_system_prompt(
        "You are a helpful assistant that describes images in detail. "
        "Focus on the visual content, structure, and key elements. "
        "Do not mention that you are looking at an image. "
        "Provide a clear, concise description of what you see."
    )

    user_prompt = (
        "Describe this image in detail. Focus on:\n"
        "- What objects, diagrams, or elements are visible\n"
        "- The layout and structure\n"
        "- Any text content (describe it, don't transcribe verbatim)\n"
        "- Colors, shapes, and relationships between elements\n"
        "- The overall purpose or meaning of the visual content"
    )

    if source_id:
        user_prompt += f"\n\nContext: This is from source '{source_id}'"

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
                'temperature': temperature,
                'num_predict': 512,  # Limit response length
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

    # Create synthesis prompt (universal guidelines + task-specific role)
    system_prompt = compose_system_prompt(
        "You are a helpful assistant that creates cohesive video summaries. "
        "Combine visual descriptions and audio content into a single, flowing narrative."
    )

    user_prompt = (
        "Based on the following frame descriptions"
        + (" and audio transcript" if transcript else "") +
        ", create ONE cohesive summary of the entire video.\n\n"
        "Guidelines:\n"
        "- Write as a continuous narrative, not a list\n"
        "- Include both visual and audio elements\n"
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

        # Categorize models (vision = multimodal / image-capable names Ollama commonly uses)
        _vision_markers = (
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen-vl",
            "qwen2-vl",
            "qwen2.5-vl",
            "llama3.2-vision",
            "granite3.2-vision",
            "pixtral",
            "internvl",
            "glm-4v",
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


def format_multi_source_context(sources: list[dict[str, Any]]) -> str:
    """
    Build a single text block from processed sources for LLM context.

    Each source dict should include:
      - 'name': str (display filename)
      - 'segments': list[tuple[str, str]] (label, visual summary text)
      - 'transcript': optional str (e.g. video audio)
    """
    blocks: list[str] = []
    for src in sources:
        name = str(src.get("name", "unknown"))
        blocks.append(f"=== SOURCE: {name} ===")
        for label, summ in src.get("segments") or []:
            blocks.append(f"--- {label} ---\n{str(summ).strip()}")
        tr = src.get("transcript")
        if tr and str(tr).strip():
            blocks.append(f"--- AUDIO TRANSCRIPT ({name}) ---\n{str(tr).strip()}")
    return "\n\n".join(blocks)


def synthesize_multi_source_corpus(
    sources: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.3,
) -> str:
    """
    One cohesive summary across multiple uploaded documents/media.

    Each item in ``sources`` matches :func:`format_multi_source_context`.
    """
    if not sources:
        return "No content to summarize."

    context = format_multi_source_context(sources)
    if not context.strip():
        return "No content to summarize."

    system_prompt = compose_system_prompt(
        "You synthesize content from multiple documents and/or media into one clear overview. "
        "Integrate related ideas across sources; note which source supports important claims when useful. "
        "Write connected prose; avoid bare bullet dumps unless it genuinely helps the reader."
    )

    user_prompt = (
        "The following is extracted from one or more uploads (visual descriptions per segment, "
        "plus optional audio transcripts for videos). Produce ONE unified summary that covers "
        "all material at a high level, highlighting themes, facts, and how pieces relate.\n\n"
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
                'num_predict': 2048,
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
        "If prior Q&A is included, use it only for conversational continuity — facts must still come "
        "from the document context."
    )

    # Trim extremely large blobs defensively (local context limits vary).
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
        raise RuntimeError(f"Local model failed (Q&A over corpus): {e}") from e


# Backwards compatibility - alias the local functions
visual_summary_from_image = visual_summary_from_image_local
video_summary_from_frame_summaries = video_summary_from_frame_summaries_local
