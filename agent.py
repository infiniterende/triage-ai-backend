"""
Agilance voice agent (LiveKit + OpenAI Realtime).

Run locally:   python agent.py dev
Production:    python agent.py start        (Render worker, see render.yaml)

Every patient/agent utterance is persisted (voice_sessions, voice_transcripts,
chat_sessions/messages) so the patient and clinician dashboards can show the
conversation, and when the session ends the clinical pathway engine evaluates
the transcript and stores a PathwayEvaluation linked to the patient record.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai

from api import AssistantFnc
from db import SessionLocal
from models import (
    ChatSession,
    Message,
    PathwayEvaluation,
    Patient,
    SpeakerType,
    VoiceSession,
    VoiceSessionStatus,
    VoiceTranscript,
)
from pathways import run_pathway
from pathways.api import extract_findings
from pathways.extraction import transcript_from_messages
from prompts import INSTRUCTIONS, WELCOME_MESSAGE

load_dotenv()
logger = logging.getLogger("agilance-voice")


def _message_text(msg: llm.ChatMessage) -> str:
    content: Any = msg.content
    if isinstance(content, list):
        return "\n".join("[image]" if isinstance(x, llm.ChatImage) else str(x) for x in content)
    return str(content or "")


class VoiceSessionRecorder:
    """Synchronous DB writes, called from the event loop via run_in_executor."""

    def __init__(self, room_name: str, participant_identity: str) -> None:
        self.room_name = room_name
        self.participant_identity = participant_identity
        self.turns: list[dict[str, str]] = []
        self._chat_session_id: Optional[int] = None
        self._voice_session_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        with SessionLocal() as db:
            voice = db.query(VoiceSession).filter_by(session_id=self.room_name).first()
            if voice is None:
                voice = VoiceSession(
                    session_id=self.room_name,
                    transcript="",
                    session_status=VoiceSessionStatus.active,
                )
                db.add(voice)
            chat = db.query(ChatSession).filter_by(session_id=self.room_name).first()
            if chat is None:
                chat = ChatSession(session_id=self.room_name, current_question=0, responses={})
                db.add(chat)
            db.commit()
            self._voice_session_id, self._chat_session_id = voice.id, chat.id
        logger.info("voice session %s started for %s", self.room_name, self.participant_identity)

    def record(self, speaker: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        role = "assistant" if speaker == "agent" else "user"
        self.turns.append({"role": role, "content": text})
        with SessionLocal() as db:
            db.add(
                VoiceTranscript(
                    voice_session_id=self._voice_session_id,
                    speaker=SpeakerType.agent if speaker == "agent" else SpeakerType.patient,
                    text=text,
                )
            )
            db.add(Message(session_id=self._chat_session_id, role=role, content=text))
            voice = db.get(VoiceSession, self._voice_session_id)
            if voice is not None:
                label = "Agent" if speaker == "agent" else "Patient"
                voice.transcript = (voice.transcript or "") + f"{label}: {text}\n"
            db.commit()

    def finish(self, patient_id: Optional[int]) -> None:
        """Evaluate the whole conversation with the pathway engine and store it."""
        if not self.turns:
            return
        transcript = transcript_from_messages(self.turns)
        findings = extract_findings(transcript)  # LLM when a key is set, keyword fallback otherwise
        result = run_pathway(findings)
        with SessionLocal() as db:
            if patient_id is None:
                patient = self._patient_from_findings(findings, result.risk_percent)
                db.add(patient)
                db.flush()
                patient_id = patient.id
            db.add(
                PathwayEvaluation(
                    session_id=self.room_name,
                    source="voice",
                    patient_id=patient_id,
                    primary_pathway=result.primary.pathway if result.primary else None,
                    disposition=result.disposition.level.value,
                    risk_percent=result.risk_percent,
                    result=result.to_dict(),
                )
            )
            voice = db.get(VoiceSession, self._voice_session_id)
            if voice is not None:
                voice.session_status = VoiceSessionStatus.completed
                voice.ended_at = datetime.utcnow()
                voice.assessment_complete = True
                voice.risk_score = result.risk_percent
                voice.patient_id = patient_id
            chat = db.get(ChatSession, self._chat_session_id)
            if chat is not None:
                chat.assessment_complete = True
                chat.risk_score = result.risk_percent or 0
            db.commit()
        logger.info(
            "voice session %s evaluated: %s / %s (%s%%)",
            self.room_name,
            result.primary.pathway if result.primary else "-",
            result.disposition.level.value,
            result.risk_percent,
        )

    def _patient_from_findings(self, findings, risk_percent: Optional[int]) -> Patient:
        yn = lambda v: "Yes" if v else ("No" if v is False else "Unknown")  # noqa: E731
        h, s, cp = findings.history, findings.symptoms, findings.chest_pain
        return Patient(
            name=self.participant_identity,
            age=findings.age or 0,
            gender=findings.sex or "unknown",
            phone_number=None,
            pain_quality=cp.quality or "unknown",
            location=yn(cp.location == "substernal") if cp.location else "Unknown",
            stress=yn(cp.exertional),
            sob=yn(s.dyspnea),
            hypertension=yn(h.hypertension),
            diabetes=yn(h.diabetes),
            hyperlipidemia=yn(h.hyperlipidemia),
            smoking=yn(h.smoking),
            probability=int(risk_percent or 0),
        )


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    loop = asyncio.get_running_loop()
    recorder = VoiceSessionRecorder(ctx.room.name, participant.identity)
    try:
        await loop.run_in_executor(None, recorder.start)
    except Exception:  # never let persistence problems stop the call
        logger.exception("could not start voice session record")

    assistant_fnc = AssistantFnc(room_name=ctx.room.name, participant_name=participant.identity)

    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice="shimmer",
        temperature=0.7,
        modalities=["audio", "text"],
        # Whisper transcription of the patient's audio is what the web app's
        # live transcript panel displays (forwarded by MultimodalAgent).
        input_audio_transcription=openai.realtime.InputTranscriptionOptions(model="whisper-1"),
    )
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)

    def _persist(speaker: str, msg: llm.ChatMessage) -> None:
        text = _message_text(msg)
        fut = loop.run_in_executor(None, recorder.record, speaker, text)
        fut.add_done_callback(
            lambda f: f.exception() and logger.error("persist failed: %s", f.exception())
        )

    @assistant.on("user_speech_committed")
    def _on_user(msg: llm.ChatMessage) -> None:
        _persist("patient", msg)

    @assistant.on("agent_speech_committed")
    def _on_agent(msg: llm.ChatMessage) -> None:
        _persist("agent", msg)

    async def _finish() -> None:
        try:
            await loop.run_in_executor(None, recorder.finish, assistant_fnc.patient_id)
        except Exception:
            logger.exception("could not finalise voice session")

    ctx.add_shutdown_callback(_finish)

    assistant.start(ctx.room, participant)

    session = model.sessions[0]
    session.conversation.item.create(llm.ChatMessage(role="assistant", content=WELCOME_MESSAGE))
    session.response.create()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
