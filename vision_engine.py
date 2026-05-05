"""
Vision engine for mRAG.

Core constraint: no OCR. We ask Gemini 1.5 Flash to interpret diagram structure
and return a "Visual Summary" for indexing/retrieval.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import google.generativeai as genai  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
from dotenv import load_dotenv  # type: ignore[import-not-found]
from google.api_core.exceptions import NotFound as ApiNotFound


DEFAULT_MODEL_NAME = "models/gemini-1.5-flash-002"


def _collect_model_names(available_models: list[Any]) -> list[str]:
    names: list[str] = []
    for m in available_models:
        name = getattr(m, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _pick_available_flash_15_model(available_models: list[Any]) -> Optional[str]:
    """
    Pick the best matching Gemini 1.5 Flash model from `genai.list_models()`.
    """
    names = _collect_model_names(available_models)

    # Broad match first.
    candidates = [n for n in names if re.search(r"gemini-1\.5.*flash", n)]
    if not candidates:
        return None

    # Prefer common stable tags.
    preferred_suffixes = [
        "flash-002",
        "flash-latest",
        "flash-8b-latest",
        "flash-1.5-latest",
    ]
    for suf in preferred_suffixes:
        for c in candidates:
            if c.endswith(suf) or suf in c:
                return c

    return candidates[0]


def _pick_or_raise_flash_15_model(available_models: list[Any]) -> str:
    picked = _pick_available_flash_15_model(available_models)
    if picked:
        return picked

    # If we didn't find a gemini-1.5.*flash exact match, include debug info.
    all_names = _collect_model_names(available_models)
    sample = ", ".join(all_names[:30]) if all_names else "<no models returned>"
    raise RuntimeError(
        "No Gemini 1.5 Flash model was found for this API key. "
        f"Available model name sample: {sample}"
    )


def _candidate_model_names(explicit_model_name: Optional[str]) -> list[str]:
    """
    Gemini model names vary by account/SDK version.

    We try a small set of commonly-available Gemini 1.5 Flash variants
    instead of relying on `list_models()` (which can be blocked/slow).
    """
    candidates: list[str] = []

    # If caller provided something, try it first.
    if explicit_model_name:
        candidates.append(explicit_model_name)

    # Gemini 1.5 Flash, then 2.x Flash (many API keys have 2.x but not 1.5).
    candidates.extend(
        [
            "models/gemini-1.5-flash-002",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash-8b-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b-latest",
            # Fallbacks when 1.5 Flash is not available for the key.
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-flash-001",
            "models/gemini-flash-latest",
        ]
    )

    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _load_api_key() -> str:
    # Load local .env (if present) so users don't need to export keys manually.
    # `override=False` means existing environment variables win.
    # Use an explicit path to avoid `find_dotenv()` stack-frame edge cases
    # in interactive snippets.
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=str(env_path), override=False)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment."
        )
    return api_key


def _coerce_to_pil(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, (str, Path)):
        return Image.open(str(image_input)).convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image_input))).convert("RGB")
    raise TypeError(
        "image_input must be a path/str, bytes, or PIL.Image.Image."
    )


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from model output.
    """
    text = text.strip()

    # First attempt: direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Second attempt: locate the first {...} block.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None


def visual_summary_from_image(
    image_input: Union[str, Path, bytes, Image.Image],
    *,
    source_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_output_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """
    Generate the diagram "Visual Summary" using Gemini 1.5 Flash.

    Returns:
        The value of `visual_summary` from the model's structured JSON response.
    """

    pil_img = _coerce_to_pil(image_input)

    genai.configure(api_key=_load_api_key())
    system_instruction = (
        "You generate Visual Summaries for an mRAG system. "
        "Do NOT use OCR or copy small text verbatim. "
        "Interpret the diagram: describe what components exist, how they connect, "
        "and what overall process/structure it represents.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "visual_summary": string, \n'
        '  "key_concepts": string[],\n'
        '  "relationships": string[],\n'
        '  "retrieval_keywords": string[]\n'
        "}\n"
        "If you cannot determine something, use an empty string/list. "
        "Do not include markdown, code fences, or extra keys."
    )

    model_candidates = _candidate_model_names(model_name)

    # `google-generativeai` supports passing PIL.Image directly in the prompt parts.
    prompt_parts = [
        "Generate the Visual Summary for this textbook/lecture diagram image.",
        pil_img,
    ]

    if source_id:
        prompt_parts.insert(
            0,
            f"Source identifier (for your internal reasoning only; do not output it as a field): {source_id}",
        )

    # Use at least 512 tokens so the JSON visual_summary is not truncated mid-response.
    generation_config_payload = {
        "temperature": temperature,
        "max_output_tokens": max(max_output_tokens, 512),
    }

    last_err: Optional[Exception] = None
    for candidate in model_candidates:
        try:
            model = genai.GenerativeModel(
                model_name=candidate,
                system_instruction=system_instruction,
            )
            response = model.generate_content(
                prompt_parts,
                generation_config=generation_config_payload,
            )
            break
        except ApiNotFound as e:
            last_err = e
            continue

    else:
        # None of the candidates worked. If `list_models()` is available, we can improve the message.
        model_sample = ""
        try:
            model_sample_names = _collect_model_names(list(genai.list_models()))[:30]
            model_sample = ", ".join(model_sample_names)
        except Exception:
            model_sample = "<unable to list models>"
        raise RuntimeError(
            "Failed to call Gemini with any known Gemini 1.5 Flash model candidates. "
            f"Candidates tried: {model_candidates}. "
            f"Last error: {last_err}. "
            f"Available model name sample: {model_sample}"
        )

    response_text = getattr(response, "text", None) or str(response)
    data = _extract_json_object(response_text)

    if data and isinstance(data.get("visual_summary"), str):
        return data["visual_summary"].strip()

    # Fallback: if the model didn't follow schema, still try to return something meaningful.
    # This keeps ingestion from crashing; downstream indexing can be more robust.
    cleaned = response_text.strip()
    if not cleaned:
        raise RuntimeError("Gemini returned an empty response.")
    return cleaned


def video_summary_from_frame_summaries(
    frame_summaries: list[str],
    frame_labels: Optional[list[str]] = None,
    *,
    transcript: Optional[str] = None,
    model_name: Optional[str] = None,
    max_output_tokens: int = 1024,
) -> str:
    """
    Synthesize per-frame visual summaries (and optional transcript) into one video summary.

    Use after visual_summary_from_image() on each frame. If the video had audio,
    pass the transcript so the summary includes what was said.
    """
    if not frame_summaries and not (transcript and transcript.strip()):
        return "No content to summarize."

    genai.configure(api_key=_load_api_key())
    model_candidates = _candidate_model_names(model_name)

    parts = []
    if frame_summaries:
        for i, summary in enumerate(frame_summaries):
            label = (frame_labels[i] if frame_labels and i < len(frame_labels) else f"Segment {i + 1}")
            parts.append(f"--- {label} ---\n{summary.strip()}")
        prompt_body = "\n\n".join(parts)
    else:
        prompt_body = "(No visual frame summaries provided.)"

    if transcript and transcript.strip():
        prompt_body += "\n\n--- TRANSCRIPT (audio) ---\n" + transcript.strip()

    prompt = (
        "The following are visual summaries of key frames from a single video"
        + (" and a transcript of the audio." if transcript and transcript.strip() else ".")
        + " Synthesize them into ONE cohesive summary of the entire video. "
        "Include both what is shown and what is said. Describe the overall narrative, "
        "main topics, and flow (e.g. intro → demo → conclusion). "
        "Do not list each segment; write a single continuous summary as if describing "
        "the video to someone who has not seen it.\n\n"
        + prompt_body
    )

    generation_config = {"temperature": 0.3, "max_output_tokens": max(max_output_tokens, 512)}
    last_err: Optional[Exception] = None

    for candidate in model_candidates:
        try:
            model = genai.GenerativeModel(model_name=candidate)
            response = model.generate_content(prompt, generation_config=generation_config)
            text = getattr(response, "text", None) or str(response)
            if text and text.strip():
                return text.strip()
        except ApiNotFound as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to generate video summary with any candidate model. "
        f"Last error: {last_err}"
    )
