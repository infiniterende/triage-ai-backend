"""
Voice / audio endpoints (FastAPI). Replaces the old Flask ``voice.py`` and the
pre-1.0 OpenAI calls that used to live in main.py.

POST /api/transcribe          audio file → transcript (OpenAI Whisper API)
POST /api/triage              audio file (+ conversation_context) → transcript + pathway result
GET  /api/questions           the scripted assessment questions
POST /voice/transcribe        alias of /api/transcribe
POST /voice/generate-speech   {text} → audio/mpeg (OpenAI TTS)
POST /voice/chat              {responses:[{role,content}]} → next assistant turn
POST /assess                  {responses:[{question, answer}]} → pathway result

Transcription is done with OpenAI's hosted Whisper model, not a local one,
so the service needs no torch/ffmpeg and starts in seconds.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from pathways import run_pathway
from pathways.extraction import extract_with_keywords, transcript_from_messages
from prompts import ASSESSMENT_QUESTIONS

router = APIRouter(tags=["voice"])

_client = None


def get_openai():
    """Lazy OpenAI client; 503 instead of a crash when no key is configured."""
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
        from openai import OpenAI

        _client = OpenAI(api_key=key)
    return _client


async def _transcribe_upload(audio: UploadFile) -> str:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        with open(path, "rb") as f:
            text = get_openai().audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                temperature=0.0,
                prompt="Medical consultation about chest pain symptoms.",
                response_format="text",
            )
    finally:
        os.remove(path)
    transcript = (text if isinstance(text, str) else getattr(text, "text", "")).strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="No speech detected in audio")
    return transcript


# --------------------------------------------------------------------------- #

@router.post("/api/transcribe")
@router.post("/voice/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> Dict[str, str]:
    transcript = await _transcribe_upload(audio)
    return {"transcript": transcript, "transcription": transcript}


@router.post("/api/triage")
async def triage_audio(
    audio: UploadFile = File(...),
    conversation_context: str = Form("{}"),
) -> Dict[str, Any]:
    """
    Transcribe one patient utterance and evaluate the conversation so far with
    the clinical pathway engine. ``conversation_context`` is JSON
    ``{"messages": [{"sender": "user"|"agent", "text": "..."}]}``.
    """
    try:
        context = json.loads(conversation_context or "{}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="conversation_context must be JSON")

    transcript = await _transcribe_upload(audio)
    messages = list(context.get("messages", []))
    messages.append({"sender": "user", "text": transcript})
    history = [
        {"role": "assistant" if m.get("sender") == "agent" else "user", "content": m.get("text", "")}
        for m in messages
    ]
    result = run_pathway(extract_with_keywords(transcript_from_messages(history)))
    payload = result.to_dict()
    return {
        "transcript": transcript,
        "risk_percent": result.risk_percent,
        "risk_level": result.disposition.level.value,
        "is_emergency": result.disposition.level.value == "emergency",
        "recommendation": result.disposition.patient_message,
        "next_question": result.next_questions[0] if result.next_questions else None,
        "detected_factors": (result.primary.supporting if result.primary else []),
        "pathway": payload,
        "conversation_context": {"messages": messages},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/questions")
def get_questions() -> Dict[str, Any]:
    return {
        "questions": [q["question"] for q in ASSESSMENT_QUESTIONS],
        "total_questions": len(ASSESSMENT_QUESTIONS),
    }


# --------------------------------------------------------------------------- #

class SpeechRequest(BaseModel):
    text: str
    voice: str = "nova"


@router.post("/voice/generate-speech")
def generate_speech(body: SpeechRequest) -> Response:
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text is required")
    audio = get_openai().audio.speech.create(model="tts-1", voice=body.voice, input=body.text)
    return Response(content=audio.content, media_type="audio/mpeg")


class ChatTurn(BaseModel):
    role: str
    content: str


class VoiceChatRequest(BaseModel):
    session_id: Optional[str] = None
    responses: List[ChatTurn] = []


VOICE_SYSTEM_PROMPT = (
    "You are a medical AI assistant conducting a chest pain assessment. Ask one "
    "question at a time, in order:\n"
    + "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(ASSESSMENT_QUESTIONS))
    + "\nKeep questions brief and clear. Never diagnose. If the patient describes "
    "crushing chest pain, breathlessness or fainting, tell them to call 911."
)


@router.post("/voice/chat")
def voice_chat(body: VoiceChatRequest) -> Dict[str, str]:
    messages = [{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
    messages += [
        {"role": t.role, "content": t.content}
        for t in body.responses
        if t.role in ("user", "assistant")
    ]
    completion = get_openai().chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=150, temperature=0.7
    )
    return {"response": (completion.choices[0].message.content or "").strip()}


# --------------------------------------------------------------------------- #

class QuestionAnswer(BaseModel):
    question_id: Optional[int] = None
    question: str
    answer: str
    session_id: Optional[str] = None


class AssessRequest(BaseModel):
    session_id: Optional[str] = None
    responses: List[QuestionAnswer]


@router.post("/assess")
def assess(body: AssessRequest) -> Dict[str, Any]:
    """Evaluate a completed question/answer set with the pathway engine."""
    if not body.responses:
        raise HTTPException(status_code=422, detail="responses is required")
    history: List[Dict[str, str]] = []
    for qa in body.responses:
        history.append({"role": "assistant", "content": qa.question})
        history.append({"role": "user", "content": qa.answer})
    result = run_pathway(extract_with_keywords(transcript_from_messages(history)))
    return {
        "session_id": body.session_id,
        "risk_percent": result.risk_percent,
        "risk_level": result.disposition.level.value,
        "recommendation": result.disposition.patient_message,
        "pathway": result.to_dict(),
    }
