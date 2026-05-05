"""
Universal system-level guidelines for every LLM call (Ollama, Gemini, etc.).

How to customize
----------------
1. Edit ``DEFAULT_UNIVERSAL_SYSTEM_GUIDELINES`` in this file (team defaults in git).
2. Or set environment variable ``SECONDBRAIN_UNIVERSAL_SYSTEM_PROMPT`` to the full
   universal block (overrides the default string; still composed with task-specific text).
3. Or set ``SECONDBRAIN_UNIVERSAL_SYSTEM_PROMPT_FILE`` to a UTF-8 text file path whose
   contents replace the default (ignored if ``SECONDBRAIN_UNIVERSAL_SYSTEM_PROMPT`` is set).

``compose_system_prompt`` prepends these guidelines to each route's own system text.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

_ENV_PROMPT = "SECONDBRAIN_UNIVERSAL_SYSTEM_PROMPT"
_ENV_PROMPT_FILE = "SECONDBRAIN_UNIVERSAL_SYSTEM_PROMPT_FILE"

DEFAULT_UNIVERSAL_SYSTEM_GUIDELINES = """\
You are a neutral analyst. Reason only from the user-supplied images, text, or transcripts—
not from who is asking, what UI they use, or any product or app name unless that name appears
verbatim in the supplied material.
For every response:
- Ground claims in the provided image, transcript, or text; do not invent facts, titles, or brands.
- Never attribute the content to a specific application, assistant product name, or codebase name
  unless that string appears in the supplied material itself.
- Prefer explaining over only describing: say what things mean, how they connect, and why
  they matter for the topic—not just what appears on the surface. Use how/why/when the
  content supports it; avoid flat inventories unless the task explicitly needs a list.
- Organize as a short, coherent account (flow from idea to idea) rather than disconnected
  observations; use light signposting (e.g. first, then, therefore) where it clarifies logic.
- Be clear and concise; avoid filler and unnecessary hedging.
- When structure helps (lists, short paragraphs), use it; otherwise use plain prose.
"""


def _load_dotenv_once() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=str(env_path), override=False)


def get_universal_system_guidelines() -> str:
    """
    Return the universal system block (trimmed), with env/file overrides applied.
    """
    _load_dotenv_once()
    raw = os.getenv(_ENV_PROMPT, "").strip()
    if raw:
        return raw

    path_str = os.getenv(_ENV_PROMPT_FILE, "").strip()
    if path_str:
        p = Path(path_str).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()

    return DEFAULT_UNIVERSAL_SYSTEM_GUIDELINES.strip()


def compose_system_prompt(*task_specific_parts: str, separator: str = "\n\n---\n\n") -> str:
    """
    Build a full system message: universal guidelines, then each non-empty task-specific part.
    """
    universal = get_universal_system_guidelines()
    tail = [p.strip() for p in task_specific_parts if p and p.strip()]
    if not tail:
        return universal
    return separator.join([universal, *tail])
