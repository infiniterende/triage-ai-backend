"""
Agilance voice agent (LiveKit Agents 1.x + OpenAI Realtime, GA API).

Run locally:   python agent.py dev
Production:    python agent.py start        (Render background worker)

Parity with the text assessment:
* the patient's speech is transcribed by OpenAI inside the realtime session
  and forwarded to the web app, so the live transcript fills in as they talk;
* after every patient answer the clinical pathway engine re-evaluates the
  conversation and the running estimate is streamed to the web app on the
  ``agilance.pathway`` text-stream topic (``final: false``);
* when the agent saves the assessment the full result is persisted and
  streamed (``final: true``) — the same PathwayResultCard the text chat shows;
* every utterance is persisted (voice_sessions, voice_transcripts,
  chat_sessions/messages) and, if the patient hangs up before the agent saved
  the assessment, the transcript is evaluated at shutdown instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobExecutorType,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    UserInputTranscribedEvent,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai

from api import PatientTools
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
from pathways.engine import PathwayResult
from pathways.extraction import extract_with_keywords, transcript_from_messages
from prompts import INSTRUCTIONS, WELCOME_MESSAGE

load_dotenv()
logger = logging.getLogger("agilance-voice")

PATHWAY_TOPIC = "agilance.pathway"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

class VoiceSessionRecorder:
    """Synchronous DB writes; the agent calls these through asyncio.to_thread."""

    def __init__(self, room_name: str, participant_identity: str) -> None:
        self.room_name = room_name
        self.participant_identity = participant_identity
        self.turns: list[dict[str, str]] = []
        self._chat_session_id: Optional[int] = None
        self._voice_session_id: Optional[int] = None

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
        text = (text or "").strip()
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

    def interim_result(self) -> Optional[PathwayResult]:
        """Cheap keyword-based evaluation of the conversation so far."""
        if not any(t["role"] == "user" for t in self.turns):
            return None
        return run_pathway(extract_with_keywords(transcript_from_messages(list(self.turns))))

    def finish(self, patient_id: Optional[int], evaluation_id: Optional[int]) -> Optional[PathwayResult]:
        """
        Close the session. If the agent already saved an evaluation during the
        call, just mark things complete; otherwise evaluate the whole
        transcript now (LLM extraction when a key is configured).
        """
        if not self.turns:
            return None
        result: Optional[PathwayResult] = None
        with SessionLocal() as db:
            if evaluation_id is None:
                transcript = transcript_from_messages(self.turns)
                result = run_pathway(extract_findings(transcript))
                if patient_id is None:
                    patient = self._patient_from_findings(result.findings, result.risk_percent)
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
                risk = result.risk_percent
            else:
                stored = db.get(PathwayEvaluation, evaluation_id)
                risk = stored.risk_percent if stored else None
            voice = db.get(VoiceSession, self._voice_session_id)
            if voice is not None:
                voice.session_status = VoiceSessionStatus.completed
                voice.ended_at = datetime.utcnow()
                voice.assessment_complete = True
                voice.risk_score = risk
                voice.patient_id = patient_id
            chat = db.get(ChatSession, self._chat_session_id)
            if chat is not None:
                chat.assessment_complete = True
                chat.risk_score = risk or 0
            db.commit()
        logger.info("voice session %s closed (risk %s%%, evaluation %s)", self.room_name, risk, evaluation_id or "from transcript")
        return result

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


# --------------------------------------------------------------------------- #
# Streaming the pathway result to the web app
# --------------------------------------------------------------------------- #

def pathway_payload(result: PathwayResult, *, final: bool, patient_id: Optional[int], evaluation_id: Optional[int]) -> Dict[str, Any]:
    return {
        "type": "pathway",
        "final": final,
        "patient_id": patient_id,
        "evaluation_id": evaluation_id,
        "result": result.to_dict(),
    }


async def publish_pathway(room: rtc.Room, payload: Dict[str, Any]) -> None:
    try:
        await room.local_participant.send_text(json.dumps(payload), topic=PATHWAY_TOPIC)
    except Exception:
        logger.exception("could not publish pathway result to the room")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class TriageAgent(Agent):
    def __init__(self, tools: PatientTools, room: rtc.Room) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._patient_tools = tools
        self._room = room

    @function_tool()
    async def save_patient_assessment(
        self,
        context: RunContext,
        name: str,
        age: int,
        sex: str,
        phone_number: str,
        pain_quality: str,
        substernal: bool,
        radiates_to_arm_or_jaw: bool,
        exertional: bool,
        relieved_by_rest: bool,
        ongoing: bool,
        shortness_of_breath: bool,
        sweating: bool,
        nausea: bool,
        hypertension: bool,
        diabetes: bool,
        hyperlipidemia: bool,
        smoking: bool,
        heart_disease: bool,
    ) -> str:
        """Save the patient's chest pain assessment once you have their answers,
        and get the risk level and recommendation to read back to them.

        Args:
            name: Patient's name.
            age: Age in years.
            sex: "male" or "female".
            phone_number: Phone number, or an empty string.
            pain_quality: pressure, sharp, burning, tearing or dull.
            substernal: Pain in the centre of the chest / behind the breastbone.
            radiates_to_arm_or_jaw: Pain spreads to the arm, jaw or neck.
            exertional: Brought on by physical activity or stress.
            relieved_by_rest: Eases within minutes of resting.
            ongoing: The pain is happening right now.
            shortness_of_breath: Short of breath.
            sweating: Sweating or clammy with the pain.
            nausea: Nausea or vomiting.
            hypertension: History of high blood pressure.
            diabetes: History of diabetes.
            hyperlipidemia: History of high cholesterol.
            smoking: Current or past smoker.
            heart_disease: Known heart disease, prior heart attack, stent or bypass.
        """
        message = await asyncio.to_thread(
            self._patient_tools.save_patient_assessment,
            name=name, age=age, sex=sex, phone_number=phone_number, pain_quality=pain_quality,
            substernal=substernal, radiates_to_arm_or_jaw=radiates_to_arm_or_jaw,
            exertional=exertional, relieved_by_rest=relieved_by_rest, ongoing=ongoing,
            shortness_of_breath=shortness_of_breath, sweating=sweating, nausea=nausea,
            hypertension=hypertension, diabetes=diabetes, hyperlipidemia=hyperlipidemia,
            smoking=smoking, heart_disease=heart_disease,
        )
        tools = self._patient_tools
        if tools.last_result is not None:
            await publish_pathway(
                self._room,
                pathway_payload(tools.last_result, final=True, patient_id=tools.patient_id, evaluation_id=tools.evaluation_id),
            )
        return message


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    recorder = VoiceSessionRecorder(ctx.room.name, participant.identity)
    try:
        await asyncio.to_thread(recorder.start)
    except Exception:  # never let persistence problems stop the call
        logger.exception("could not start voice session record")

    tools = PatientTools(room_name=ctx.room.name, participant_name=participant.identity)

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(model="gpt-realtime", voice="shimmer"),
    )

    def _log_failure(what: str):
        def cb(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception():
                logger.error("%s failed: %s", what, t.exception())
        return cb

    async def _persist_and_estimate(speaker: str, text: str) -> None:
        await asyncio.to_thread(recorder.record, speaker, text)
        # Running estimate after each patient answer (skip once the agent has
        # saved the real assessment — the final card is already on screen).
        if speaker == "patient" and tools.evaluation_id is None:
            interim = await asyncio.to_thread(recorder.interim_result)
            if interim is not None:
                await publish_pathway(ctx.room, pathway_payload(interim, final=False, patient_id=None, evaluation_id=None))

    @session.on("user_input_transcribed")
    def _on_user(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            asyncio.create_task(_persist_and_estimate("patient", ev.transcript)).add_done_callback(_log_failure("persist"))

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        # The event also carries non-message items (e.g. AgentHandoff) that
        # have no role/text; only persist what the agent actually said.
        item = ev.item
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            asyncio.create_task(_persist_and_estimate("agent", item.text_content or "")).add_done_callback(_log_failure("persist"))

    async def _finish() -> None:
        try:
            await asyncio.to_thread(recorder.finish, tools.patient_id, tools.evaluation_id)
        except Exception:
            logger.exception("could not finalise voice session")

    ctx.add_shutdown_callback(_finish)

    await session.start(
        agent=TriageAgent(tools, ctx.room),
        room=ctx.room,
        room_input_options=RoomInputOptions(participant_identity=participant.identity),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    await session.generate_reply(instructions=WELCOME_MESSAGE)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Run jobs as threads in this process: on a 0.5 CPU / 512 MB worker a
            # second interpreter importing the runtime blew the memory limit and
            # the 10 s process-init deadline, so no room was ever served.
            job_executor_type=JobExecutorType.THREAD,
            num_idle_processes=1,
            initialize_process_timeout=120.0,
        )
    )
