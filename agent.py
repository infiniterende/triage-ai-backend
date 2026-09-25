"""
Agilance voice agent (LiveKit Agents 1.x + OpenAI Realtime, GA API).

Run locally:   python agent.py dev
Production:    python agent.py start        (Render background worker)

The patient's speech is transcribed by OpenAI inside the realtime session and
forwarded to the web app (legacy transcription events and lk.transcription
text streams), so the browser's live transcript panel fills in as they talk.
Every utterance is persisted (voice_sessions, voice_transcripts,
chat_sessions/messages) and, when the call ends, the clinical pathway engine
evaluates the transcript and stores a PathwayEvaluation linked to the patient.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    JobExecutorType,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
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
from pathways.extraction import transcript_from_messages
from prompts import INSTRUCTIONS, WELCOME_MESSAGE

load_dotenv()
logger = logging.getLogger("agilance-voice")


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


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class TriageAgent(Agent):
    def __init__(self, tools: PatientTools) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._patient_tools = tools

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
        return await asyncio.to_thread(
            self._patient_tools.save_patient_assessment,
            name=name, age=age, sex=sex, phone_number=phone_number, pain_quality=pain_quality,
            substernal=substernal, radiates_to_arm_or_jaw=radiates_to_arm_or_jaw,
            exertional=exertional, relieved_by_rest=relieved_by_rest, ongoing=ongoing,
            shortness_of_breath=shortness_of_breath, sweating=sweating, nausea=nausea,
            hypertension=hypertension, diabetes=diabetes, hyperlipidemia=hyperlipidemia,
            smoking=smoking, heart_disease=heart_disease,
        )


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

    def _persist(speaker: str, text: str) -> None:
        task = asyncio.create_task(asyncio.to_thread(recorder.record, speaker, text))
        task.add_done_callback(
            lambda t: t.exception() and logger.error("persist failed: %s", t.exception())
        )

    @session.on("user_input_transcribed")
    def _on_user(ev: UserInputTranscribedEvent) -> None:
        if ev.is_final:
            _persist("patient", ev.transcript)

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.role == "assistant":
            _persist("agent", ev.item.text_content or "")

    async def _finish() -> None:
        try:
            await asyncio.to_thread(recorder.finish, tools.patient_id)
        except Exception:
            logger.exception("could not finalise voice session")

    ctx.add_shutdown_callback(_finish)

    await session.start(
        agent=TriageAgent(tools),
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
