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

        # Categorize models
        vision_models = [name for name in model_names if any(v in name.lower() for v in ['llava', 'bakllava', 'moondream'])]
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



# Backwards compatibility - alias the local functions
visual_summary_from_image = visual_summary_from_image_local
video_summary_from_frame_summaries = video_summary_from_frame_summaries_local
