"""
Persist chat sessions / “interactions” locally (JSON under data/interactions/).

One interaction = one completed summarize run (uploaded media) + optional Ask/Q&A thread.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

INTERACTIONS_DIR = Path("data") / "interactions"
SCHEMA_VERSION = 1


def ensure_dir() -> None:
    INTERACTIONS_DIR.mkdir(parents=True, exist_ok=True)


def _path_for_id(iid: str) -> Path:
    return INTERACTIONS_DIR / f"{iid}.json"


def serialize_batch_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in sources:
        d = dict(src)
        d["segments"] = [[a, b] for a, b in (src.get("segments") or [])]
        vs = src.get("verbatim_segments")
        d["verbatim_segments"] = [[a, b] for a, b in (vs or [])]
        out.append(d)
    return out


def deserialize_batch_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for src in sources:
        d = dict(src)
        d["segments"] = [tuple(x) for x in (src.get("segments") or [])]
        d["verbatim_segments"] = [tuple(x) for x in (src.get("verbatim_segments") or [])]
        out.append(d)
    return out


def _title_from_batch_meta(meta: list[dict[str, Any]]) -> str:
    if not meta:
        return "Session"
    names = [str(m.get("name", "")) for m in meta if m.get("name")]
    if not names:
        return "Session"
    head = ", ".join(names[:2])
    if len(names) > 2:
        head += f" (+{len(names) - 2} more)"
    return head[:200]


def new_interaction_record(
    *,
    batch_sources: list[dict[str, Any]],
    corpus_summary: Optional[str],
    corpus_context: Optional[str],
    source_batch: list[dict[str, Any]],
    text_model: str,
    vision_model: str,
    chat_messages: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    iid = str(uuid.uuid4())
    return {
        "id": iid,
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "title": _title_from_batch_meta(source_batch),
        "batch_sources": serialize_batch_sources(batch_sources),
        "corpus_summary": corpus_summary,
        "corpus_context": corpus_context,
        "source_batch": source_batch,
        "text_model": text_model,
        "vision_model": vision_model,
        "chat_messages": chat_messages or [],
    }


def save_interaction(record: dict[str, Any]) -> None:
    ensure_dir()
    iid = record.get("id")
    if not iid:
        raise ValueError("interaction record must have id")
    with open(_path_for_id(str(iid)), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def load_interaction(iid: str) -> Optional[dict[str, Any]]:
    p = _path_for_id(iid)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def list_interactions() -> list[dict[str, Any]]:
    ensure_dir()
    rows: list[dict[str, Any]] = []
    for p in sorted(INTERACTIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                rows.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def update_interaction_chat(iid: str, chat_messages: list[dict[str, str]]) -> None:
    rec = load_interaction(iid)
    if not rec:
        return
    rec["chat_messages"] = list(chat_messages)
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_interaction(rec)


def apply_loaded_interaction(rec: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs to merge into Streamlit session_state for Ask + summaries."""
    out: dict[str, Any] = {
        "batch_sources": deserialize_batch_sources(rec.get("batch_sources") or []),
        "corpus_summary": rec.get("corpus_summary"),
        "corpus_context": rec.get("corpus_context"),
        "chat_messages": list(rec.get("chat_messages") or []),
        "active_interaction_id": rec.get("id"),
        "stage": "done",
    }
    sb = rec.get("source_batch") or []
    if sb:
        out["media_path"] = sb[0].get("media_path")
        out["is_video"] = bool(sb[0].get("is_video"))
    return out
