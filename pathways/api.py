"""
HTTP surface for the pathway engine. Mount with::

    from pathways.api import router as pathway_router
    app.include_router(pathway_router)

Endpoints
---------
POST /api/pathway/evaluate         findings JSON → full pathway result
POST /api/pathway/from-transcript  chat messages → extraction → pathway result
GET  /api/pathway/catalog          pathway + disposition definitions for the UI
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .conditions import ACUTE_PATHWAYS, CHRONIC_PATHWAYS
from .disposition import DISPOSITION_META, Disposition
from .engine import ENGINE_VERSION, evaluate, run_pathway
from .extraction import extract_with_keywords, extract_with_llm, transcript_from_messages
from .findings import ClinicalFindings

router = APIRouter(prefix="/api/pathway", tags=["clinical-pathways"])


class EvaluateRequest(BaseModel):
    findings: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: str


class TranscriptRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = None
    transcript: Optional[str] = None
    use_llm: bool = True
    model: str = "gpt-4.1-mini"


def _openai_client():
    """Lazily create the OpenAI client so the engine works without a key."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI  # type: ignore

        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:  # pragma: no cover
        return None


def extract_findings(transcript: str, use_llm: bool = True, model: str = "gpt-4.1-mini") -> ClinicalFindings:
    client = _openai_client() if use_llm else None
    if client is None:
        return extract_with_keywords(transcript)
    try:
        return extract_with_llm(transcript, client, model=model)
    except Exception:
        return extract_with_keywords(transcript)


@router.post("/evaluate")
def evaluate_findings(body: EvaluateRequest) -> Dict[str, Any]:
    return evaluate(body.findings)


@router.post("/from-transcript")
def evaluate_transcript(body: TranscriptRequest) -> Dict[str, Any]:
    if not body.messages and not body.transcript:
        raise HTTPException(status_code=422, detail="Provide `messages` or `transcript`.")
    transcript = body.transcript or transcript_from_messages([m.dict() for m in body.messages or []])
    findings = extract_findings(transcript, use_llm=body.use_llm, model=body.model)
    result = run_pathway(findings)
    return {"transcript": transcript, "result": result.to_dict()}


@router.get("/catalog")
def pathway_catalog() -> Dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "pipeline": [
            "Voice / text conversation",
            "Symptom + history extraction",
            "Safety / red-flag layer",
            "Cardiovascular triage router",
            "Condition-specific assessment",
            "Disposition recommendation",
        ],
        "acute_pathways": [
            {"id": m.ID, "name": m.NAME, "description": m.DESCRIPTION} for m in ACUTE_PATHWAYS
        ],
        "chronic_pathways": [
            {
                "id": getattr(m, "CHRONIC_ID", m.ID),
                "name": getattr(m, "CHRONIC_NAME", m.NAME),
                "description": m.DESCRIPTION,
            }
            for m in CHRONIC_PATHWAYS
        ],
        "dispositions": [
            {
                "level": level.value,
                "headline": meta["headline"],
                "timeframe": meta["timeframe"],
                "scheduling": meta["scheduling"],
            }
            for level, meta in DISPOSITION_META.items()
        ],
    }
