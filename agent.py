from __future__ import annotations
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from dotenv import load_dotenv
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS
import os
from models import ChatSession, Message
from livekit.plugins import openai
import asyncio

from db import SessionLocal

db = SessionLocal()
from datetime import datetime


async def handle_user_message(msg: llm.ChatMessage, assistant_fnc: AssistantFnc):
    with SessionLocal() as db:
        existing_sessions = db.query(ChatSession).count()
        session_id = f"session_{existing_sessions + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Retrieve or create ChatSession
        chat_session = (
            db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        )
        if not chat_session:
            chat_session = ChatSession(
                session_id=session_id,
                current_question=0,
                responses={},
                risk_score=0,
                assessment_complete=False,
                messages=[],
                conversation_history=[],
            )
            db.add(chat_session)

        # Create user message
        user_msg = Message(session_id=session_id, role="user", content=msg.content)
        db.add(user_msg)

        # Update session state
        chat_session.messages.append(user_msg)
        chat_session.conversation_history.append(user_msg)
        chat_session.current_question += 1

        # Complete assessment if finished
        if chat_session.current_question >= 10 and not chat_session.assessment_complete:
            chat_session.assessment_complete = True
            conversation_history = [
                m.content for m in chat_session.conversation_history
            ]
            patient_data = await assistant_fnc.extract_patient_data(
                conversation_history
            )
            await assistant_fnc.create_patient(**patient_data)

        db.commit()
        db.refresh(chat_session)

        voice_session = (
            db.query(VoiceSession)
            .filter_by(session_id=assistant_fnc.voice_session_id)
            .first()
        )

        if not voice_session:
            voice_session = VoiceSession(
                session_id=assistant_fnc.voice_session_id,
                status=VoiceSessionStatus.active,
            )
            db.add(voice_session)
            db.commit()
            db.refresh(voice_session)
            transcript = VoiceTranscript(
                voice_session_id=voice_session.id,
                speaker=SpeakerType.patient,
                text=msg.content,
            )
            db.add(transcript)
            db.commit()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    await ctx.wait_for_participant()
    voice_session_id = ctx.room.name
    assistant_fnc.voice_session_id = voice_session_id
    assistant_fnc.room_name = ctx.room.name
    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice="shimmer",
        temperature=0.8,
        modalities=["audio", "text"],
    )
    assistant_fnc = AssistantFnc()
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
    assistant.start(ctx.room)

    session = model.sessions[0]
    session.conversation.item.create(
        llm.ChatMessage(role="assistant", content=WELCOME_MESSAGE)
    )
    session.response.create()

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            msg.content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else x for x in msg
            )
        asyncio.create_task(handle_user_message(msg, assistant_fnc))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
