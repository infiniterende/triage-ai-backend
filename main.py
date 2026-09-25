from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Query,
    Depends,
    status,
    Security,
)

from sqlalchemy import text
from datetime import datetime
from db import engine, Base, SessionLocal

import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from prompts import ASSESSMENT_QUESTIONS

from fastapi.security import OAuth2PasswordBearer

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
from enum import Enum
import os
from openai import OpenAI
from datetime import datetime
import tempfile
import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import asyncio
import io
import uuid
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from models import Doctor, Patient, ChatSession, Message, Base
from jose import JWTError, jwt

from auth import create_access_token
from auth_api import verify_password

from livekit import api
from livekit.api import LiveKitAPI, ListRoomsRequest
from dotenv import load_dotenv

from db import engine, Base, SessionLocal, get_db

load_dotenv()

from calculate_cad_score import classify_chest_pain, cadc_clinical_risk
from models import PathwayEvaluation, Base as ModelsBase
from pathways import run_pathway
from pathways.api import extract_findings, router as pathway_router
from pathways.extraction import extract_with_keywords, transcript_from_messages
from care_api import router as care_router
from auth_api import router as auth_router
from voice_api import router as voice_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI(title="AI Health Assistant", version="1.0.0")

# Allow frontend origins
origins = [
    "https://agilance-frontend.vercel.app",
    "http://localhost:3000",  # frontend dev server
    "https://api.agilance.org",
    "https://main.d36t856vyywoj3.amplifyapp.com",  # production frontend
    "https://triage-ai-delta.vercel.app"
]


# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """
    Return JSON 500s through the middleware stack. Without this, an unhandled
    error skips CORSMiddleware and the browser reports a misleading CORS
    failure instead of the real server error.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Clinical pathway engine + care-coordination endpoints
app.include_router(pathway_router)
app.include_router(care_router)
app.include_router(auth_router)
app.include_router(voice_router)


def run_and_store_pathway(
    db: Session,
    session_id: str,
    messages_history: List[Dict[str, str]],
    patient_id: Optional[int] = None,
    source: str = "text",
    use_llm: bool = True,
):
    """
    Symptom/history extraction → red flags → router → disposition, persisted as
    a PathwayEvaluation row so dashboards can read it back later.
    """
    transcript = transcript_from_messages(messages_history)
    findings = extract_findings(transcript, use_llm=use_llm) if use_llm else extract_with_keywords(transcript)
    result = run_pathway(findings)
    payload = result.to_dict()
    evaluation = PathwayEvaluation(
        session_id=session_id,
        source=source,
        patient_id=patient_id,
        primary_pathway=result.primary.pathway if result.primary else None,
        disposition=result.disposition.level.value,
        risk_percent=result.risk_percent,
        result=payload,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    payload["evaluation_id"] = evaluation.id
    return result, payload


class UserResponse(BaseModel):
    question_id: int
    question: str
    answer: str
    session_id: str


class AssessmentRequest(BaseModel):
    responses: List[UserResponse]
    session_id: str


class AppointmentRequest(BaseModel):
    patient_name: str
    email: str
    phone: str
    preferred_date: str
    preferred_time: str
    session_id: str


class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class UserResponse(BaseModel):
    session_id: str
    message: str


# `Base` was shadowed by the empty declarative_base from db.py, so this used
# to create nothing on a fresh database. Use the models' metadata explicitly.
ModelsBase.metadata.create_all(bind=engine)

df = pd.read_csv("clean_patients.csv")


# Convert 'Yes'/'No' strings to booleans
for col in ["stress", "sob", "hypertension", "diabetes", "hyperlipidemia", "smoking"]:
    df[col] = df[col].fillna(False)  # or True depending on your app
    df[col] = df[col].astype(bool)  # cast to actual boolean type
    # df[col] = df[col].map({"Yes": True, "yes": True, "no": False, "No": False})

for col in ["age", "probability"]:
    df["age"] = df["age"].fillna(0).astype(int)
    df["probability"] = df["probability"].fillna(0).astype(int)


# ----------------------------
# 6. Seed function
# ----------------------------


def clear_db():
    db = SessionLocal()
    try:
        db.execute(
            text(
                "TRUNCATE TABLE patients, chat_sessions, messages RESTART IDENTITY CASCADE;"
            )
        )
        db.commit()
        print("✅ Database cleared")
    except Exception as e:
        db.rollback()
        print("❌ Error clearing DB:", e)
    finally:
        db.close()


def seed():
    db: Session = SessionLocal()
    try:
        patients = [
            Patient(
                name=row["name"],
                age=row["age"],
                gender=row["gender"],
                phone_number=row["phone_number"],
                pain_quality=row["pain_quality"],
                location=row["location"],
                stress=row["stress"],
                sob=row["sob"],
                hypertension=row["hypertension"],
                diabetes=row["diabetes"],
                hyperlipidemia=row["hyperlipidemia"],
                smoking=row["smoking"],
                probability=row["probability"],
            )
            for _, row in df.iterrows()
        ]
        db.add_all(patients)
        db.commit()
        print("✅ Seed data inserted into Supabase!")
    except Exception as e:
        db.rollback()
        print("❌ Error:", e)
    finally:
        db.close()


# ----------------------------
# 7. Run
# ----------------------------

# clear_db()
# seed()


# In-memory storage (use database in production)
sessions: Dict[str, ChatSession] = {}

# OpenAI System Prompt for Health Assessment
SYSTEM_PROMPT = f"""
You are a professional medical AI assistant specializing in chest pain assessment. Your role is to:

1. Conduct a structured assessment through specific questions
2. Maintain a compassionate, professional tone
3. Prioritize patient safety and encourage appropriate care-seeking behavior
4. NEVER provide definitive medical diagnoses
5. Always emphasize that this is a preliminary assessment

CRITICAL SAFETY RULES:
- If a patient mentions severe, crushing chest pain, difficulty breathing, or feels they're having a heart attack, immediately recommend calling 911
- Always remind patients this is not a substitute for professional medical care
- Be supportive but clear about limitations

When asking assessment questions:
- Be conversational but ensure you get the specific information needed for risk calculation
- Ask one question at a time
- Acknowledge the patient's previous response before moving on
- Follow the exact order of these 10 questions:

{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(ASSESSMENT_QUESTIONS)])}

"""


async def get_openai_response(
    messages_history: List[Dict[str, str]], current_question_info: Dict[str, Any]
) -> str:
    """Get response from OpenAI API"""
    try:
        # Simple request: list available models
        models = client.models.list()
        for model in models.data:
            print("-", model.id)
    except Exception as e:
        print("API key validation failed:", str(e))

    try:
        # Prepare the conversation context
        system_message = SYSTEM_PROMPT
        if current_question_info:
            system_message += (
                f"\n\nCURRENT QUESTION TO ASK: {current_question_info['question']}"
            )
            system_message += f"\nQUESTION TYPE: {current_question_info['type']}"
            system_message += (
                f"\nQUESTION NUMBER: {current_question_info['number']} of 7"
            )

        # Build the conversation
        conversation = [{"role": "system", "content": system_message}]
        conversation.extend(messages_history)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=300,
            temperature=0.7,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I'm experiencing technical difficulties. For immediate medical concerns, please contact your healthcare provider or call 911 if this is an emergency."


def calculate_risk_score(responses: Dict[str, Any]) -> int:
    """Calculate risk score based on user responses"""
    total_score = 0

    risk_factor_score = 0
    for question in ASSESSMENT_QUESTIONS:
        question_id = question["id"]
        if question_id in responses:
            response = str(responses[question_id]).lower()
            location_score = 0
            trigger_score = 0
            relief_score = 0
            age = 0
            male = 0
            diabetes_score = 0
            hypertension_score = 0
            dyslipidemia_score = 0
            smoking_score = 0
            for keyword, score in question["scoring"].items():
                if keyword in response and question_id == "location":
                    location_score = 1
                if keyword in response and question_id == "trigger":
                    trigger_score = 1
                if keyword in response and question_id == "relief":
                    relief_score = 1
            if question_id == "age":
                age = int(response)
            if question_id == "risk_factors":
                for keyword, score in question["scoring"].items():
                    if keyword in response:
                        if keyword == "diabetes":
                            diabetes_score = 1
                        if keyword == "pressure":
                            hypertension_score = 1
                        if keyword == "cholesterol":
                            dyslipidemia_score = 1
                        if keyword == "smoking":
                            smoking_score = 1
                        if keyword == "male":
                            male = 1

            chest_pain_type = classify_chest_pain(
                location_score, trigger_score, relief_score
            )
            risk_probability = cadc_clinical_risk(
                age,
                male,
                chest_pain_type,
                diabetes_score,
                hypertension_score,
                dyslipidemia_score,
                smoking_score,
            )

    return risk_probability * 100


def get_risk_level_and_recommendation(score: int) -> tuple:
    """Determine risk level and recommendation based on score"""
    if score >= 0.15:
        return (
            RiskLevel.CRITICAL,
            "🚨 SEEK EMERGENCY CARE IMMEDIATELY - Call 911 or go to the nearest emergency room right away. Your symptoms suggest a possible heart attack or other serious cardiac emergency.",
        )
    elif score > 0.05:
        return (
            RiskLevel.HIGH,
            "⚠️ HIGH RISK - You should go to the emergency room or urgent care immediately. Do not drive yourself - have someone drive you or call for emergency transport.",
        )
    elif score <= 0.05:
        return (
            RiskLevel.LOW,
            "✅ LOWER RISK - While your risk appears lower, chest pain should still be evaluated. Schedule an appointment with your healthcare provider within the next few days. Seek immediate care if symptoms worsen.",
        )


@app.post("/api/chat/start")
async def start_chat():
    """Start a new chat session"""
    db = SessionLocal()
    session_id = (
        f"session_{len(sessions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    session = ChatSession(
        session_id=session_id,
        current_question=0,
        responses={},
        risk_score=0,
        assessment_complete=False,
    )

    # Initial OpenAI conversation
    initial_context = [
        {
            "role": "user",
            "content": "I'm experiencing chest pain and would like to get an assessment.",
        }
    ]

    welcome_response = await get_openai_response(
        initial_context,
        {
            "question": ASSESSMENT_QUESTIONS[0]["question"],
            "type": ASSESSMENT_QUESTIONS[0]["type"],
            "number": 1,
        },
    )

    initial_context_msg = Message(
        session_id=session_id,
        role=MessageType.USER,
        content="I'm experiencing chest pain and would like to get an assessment.",
    )

    session.messages.append(initial_context_msg)
    session.conversation_history.append(initial_context_msg)
    welcome_msg = Message(
        session_id=session_id, role=MessageType.ASSISTANT, content=welcome_response
    )

    session.messages.append(welcome_msg)
    session.conversation_history.append(welcome_msg)

    db.add(session)
    db.commit()
    db.refresh(session)

    messages = [
        {"role": message.role, "content": message.content}
        for message in session.messages
    ]

    db.close()
    # sessions[session_id] = session
    return {"session_id": session_id, "messages": messages}


@app.post("/api/chat/message")
async def process_message(user_response: UserResponse):
    db = SessionLocal()
    """Process user message and return AI response"""
    # if user_response.session_id not in sessions:
    #     raise HTTPException(status_code=404, detail="Session not found")

    session = (
        db.query(ChatSession)
        .filter(ChatSession.session_id == user_response.session_id)
        .first()
    )
    # session = sessions[user_response.session_id]

    # Add user message to session and conversation history
    new_msg = Message(
        session_id=session.session_id,
        role=MessageType.USER,
        content=user_response.message,
    )

    session.messages.append(new_msg)

    session.conversation_history.append(new_msg)
    # session.messages.append(
    #     Message(type=MessageType.USER, content=user_response.message)
    # )
    # session.conversation_history.append(
    #     {"role": "user", "content": user_response.message}
    # )

    messages = [
        {"role": message.role, "content": message.content}
        for message in session.messages
    ]

    if session.assessment_complete:
        # Handle post-assessment conversation
        msgs = [
            {"role": msg.role, "content": msg.content}
            for msg in session.conversation_history
        ]
        print(msgs)
        response = await get_openai_response(messages, None)
        msg = Message(
            session_id=session.session_id, role=MessageType.ASSISTANT, content=response
        )
        session.messages.append(msg)
        session.conversation_history.append(msg)
        # session.messages.append(Message(type=MessageType.ASSISTANT, content=response))
        # session.conversation_history.append({"role": "assistant", "content": response})
        db.commit()
        db.refresh(session)

        msgs = [
            {"role": message.role, "content": message.content}
            for message in session.messages
        ]
        return {"messages": msgs}

    # Store response for risk calculation
    current_q = ASSESSMENT_QUESTIONS[session.current_question]
    session.responses[current_q["id"]] = user_response.message

    # Safety / red-flag layer runs on every turn (cheap keyword extraction, no
    # LLM call). If the pathway engine already sees an emergency we stop the
    # interview and tell the patient to call 911 rather than asking 6 more
    # questions.
    interim_findings = extract_with_keywords(
        transcript_from_messages(session.conversation_history)
    )
    interim = run_pathway(interim_findings)
    if interim.red_flags and interim.disposition.level.value == "emergency":
        session.assessment_complete = True
        _, pathway_payload = run_and_store_pathway(
            db, session.session_id, messages, use_llm=False
        )
        emergency_text = (
            "🚨 **Please stop and call 911 now.** "
            + " ".join(f.label + "." for f in interim.red_flags)
            + " "
            + interim.disposition.patient_message
        )
        emergency_msg = Message(
            session_id=session.session_id,
            role=MessageType.ASSISTANT,
            content=emergency_text,
        )
        session.messages.append(emergency_msg)
        session.conversation_history.append(emergency_msg)
        db.commit()
        db.refresh(session)
        msgs = [{"role": m.role, "content": m.content} for m in session.messages]
        db.close()
        return {"messages": msgs, "pathway": pathway_payload}

    # Move to next question or complete assessment
    session.current_question += 1
    print(session.current_question)
    print(session.assessment_complete)
    if session.current_question >= len(ASSESSMENT_QUESTIONS):
        # Complete assessment
        session.assessment_complete = True
        print(session.assessment_complete)

        history_serialized = [
            {"role": m.role, "content": m.content} for m in session.conversation_history
        ]
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical assistant AI that extracts structured patient data.",
                },
                {
                    "role": "user",
                    "content": f"Extract patient info as JSON with fields: name (string), age (integer), gender (string), phone_number (string), pain_quality (string), pain_location (), stress, shortness_of_breath, hypertension, diabetes, hyperlipidemia, smoking. Transcript: {history_serialized}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        patient_data = json.loads(response.choices[0].message.content)
        print(patient_data)
        risk_probability = cadc_clinical_risk(
            age=patient_data.get("age"),
            male=patient_data.get("gender"),
            chest_pain_type=classify_chest_pain(
                patient_data.get("location"),
                patient_data.get("trigger"),
                patient_data.get("relief"),
            ),
            diabetes=patient_data.get("diabetes"),
            hypertension=patient_data.get("hypertension"),
            dyslipidaemia=patient_data.get("hyperlipidemia"),
            smoking=patient_data.get("smoking"),
        )

        print(risk_probability)

        patient = Patient(
            name=patient_data.get("name"),
            age=patient_data.get("age"),
            gender=patient_data.get("gender"),
            phone_number=patient_data.get("phone_number"),
            pain_quality=patient_data.get("pain_quality"),
            location=patient_data.get("pain_location"),
            stress=patient_data.get("stress"),
            sob=patient_data.get("shortness_of_breath"),
            hypertension=patient_data.get("hypertension"),
            diabetes=patient_data.get("diabetes"),
            hyperlipidemia=patient_data.get("hyperlipidemia"),
            smoking=patient_data.get("smoking"),
            probability=risk_probability * 100,
        )

        db.add(patient)
        db.commit()
        db.refresh(patient)
        # session.risk_score = calculate_risk_score(session.responses)
        risk_level, recommendation = get_risk_level_and_recommendation(risk_probability)

        # Clinical pathway engine: full LLM extraction → red flags → router →
        # condition-specific assessment → disposition. Its output drives the
        # closing recommendation and is stored for the dashboards.
        pathway_result, pathway_payload = run_and_store_pathway(
            db, session.session_id, history_serialized, patient_id=patient.id
        )
        session.risk_score = pathway_result.risk_percent or int(risk_probability * 100)
        primary = pathway_result.primary
        red_flag_text = (
            "; ".join(f.label for f in pathway_result.red_flags) or "none identified"
        )

        print(history_serialized)
        # Get AI-generated summary and recommendation
        assessment_prompt = f"""The patient has completed the chest pain assessment. The clinical pathway engine has evaluated their answers:

CAD pre-test probability: {round(risk_probability * 100)}%
Headline risk: {pathway_result.risk_percent}%
Leading pathway: {primary.name if primary else 'none'} ({primary.likelihood.value if primary else 'n/a'})
Red flags: {red_flag_text}
Disposition: {pathway_result.disposition.level.value.upper()} — {pathway_result.disposition.headline} ({pathway_result.disposition.timeframe})
Patient guidance: {pathway_result.disposition.patient_message}
Actions: {'; '.join(pathway_result.disposition.actions)}

Based on the conversation history, provide a compassionate, concise summary of what the patient described, explain the disposition above in plain language, and state the actions clearly. Do not contradict the disposition. Do not provide a diagnosis."""
        msgs = [
            {"role": msg.role, "content": msg.content}
            for msg in session.conversation_history
        ]

        print(msgs)

        ai_response = await get_openai_response(
            msgs + [{"role": "assistant", "content": assessment_prompt}],
            None,
        )

        ai_message = Message(
            session_id=session.session_id,
            role=MessageType.ASSISTANT,
            content=ai_response,
        )

        session.messages.append(ai_message)
        session.conversation_history.append(ai_message)

        db.commit()

        db.refresh(session)

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in session.conversation_history
        ]

        return {
            "messages": messages,
            "pathway": pathway_payload,
            "patient_id": patient.id,
        }
    else:
        # Ask next question using OpenAI. The pathway engine's condition-
        # specific follow-up (if any) is offered as a hint so the interview
        # adapts to the leading pathway instead of being purely scripted.
        next_question_info = {
            "question": ASSESSMENT_QUESTIONS[session.current_question]["question"],
            "type": ASSESSMENT_QUESTIONS[session.current_question]["type"],
            "number": session.current_question + 1,
        }
        if interim.next_questions:
            next_question_info["question"] += (
                " (If it flows naturally, also ask: "
                + interim.next_questions[0]
                + ")"
            )

        ai_response = await get_openai_response(messages, next_question_info)

        ai_msg = Message(
            session_id=session.session_id,
            role=MessageType.ASSISTANT,
            content=ai_response,
        )

        session.messages.append(ai_msg)
        session.conversation_history.append(ai_msg)
        # db.add(ai_msg)

        # session.messages.append(
        #     Message(type=MessageType.ASSISTANT, content=ai_response)
        # )
        # session.conversation_history.append(
        #     {"role": "assistant", "content": ai_response}
        # )

        db.commit()
        db.refresh(session)

        session_messages = [
            {"role": message.role, "content": message.content}
            for message in session.messages
        ]
        db.close()

        return {"messages": session_messages}


@app.get("/api/patients")
async def get_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return patients


@app.get("/api/chat_sessions/")
async def get_chat_sessions(db: Session = Depends(get_db)):
    chat_sessions = db.query(ChatSession).all()
    return chat_sessions


@app.get("/api/chat/{session_id}")
async def get_chat_session(session_id: str, db: Session = Depends(get_db)):
    session = (
        db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "assessment_complete": session.assessment_complete,
        "messages": [
            {"role": m.role, "content": m.content}
            for m in sorted(session.messages, key=lambda m: m.id)
        ],
    }


# @app.get("/api/chat/{session_id}")
# async def get_chat_history(session_id: str):
#     """Get chat history for a session"""
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found")

#     return {"messages": sessions[session_id].messages}


@app.post("/api/voice")
async def process_voice(messages):
    db = SessionLocal()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant AI that extracts structured patient data.",
            },
            {
                "role": "user",
                "content": f"Extract patient info as JSON with fields: name (string), age (integer), gender (string), phone_number (string), pain_quality (string), pain_location (), stress, shortness_of_breath, hypertension, diabetes, hyperlipidemia, smoking. Transcript: {messages}",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    patient_data = json.loads(response.choices[0].message.content)
    print(patient_data)
    risk_probability = cadc_clinical_risk(
        age=patient_data.get("age"),
        male=patient_data.get("gender"),
        chest_pain_type=classify_chest_pain(
            patient_data.get("location"),
            patient_data.get("trigger"),
            patient_data.get("relief"),
        ),
        diabetes=patient_data.get("diabetes"),
        hypertension=patient_data.get("hypertension"),
        dyslipidaemia=patient_data.get("hyperlipidemia"),
        smoking=patient_data.get("smoking"),
    )

    print(risk_probability)

    patient = Patient(
        name=patient_data.get("name"),
        age=patient_data.get("age"),
        gender=patient_data.get("gender"),
        phone_number=patient_data.get("phone_number"),
        pain_quality=patient_data.get("pain_quality"),
        location=patient_data.get("pain_location"),
        stress=patient_data.get("stress"),
        sob=patient_data.get("shortness_of_breath"),
        hypertension=patient_data.get("hypertension"),
        diabetes=patient_data.get("diabetes"),
        hyperlipidemia=patient_data.get("hyperlipidemia"),
        smoking=patient_data.get("smoking"),
        probability=risk_probability * 100,
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Health Assistant"}


# Generate a unique room name that's not taken yet
async def generate_room_name():
    name = "room-" + str(uuid.uuid4())[:8]
    rooms = await get_rooms()
    while name in rooms:
        name = "room-" + str(uuid.uuid4())[:8]
    return name


# List existing room names
async def get_rooms():
    api_client = LiveKitAPI()
    rooms = await api_client.room.list_rooms(ListRoomsRequest())
    await api_client.aclose()
    return [room.name for room in rooms.rooms]


# /getToken?name=Shirley&room=my-room
@app.get("/getToken")
async def get_token(name: str = Query(...), room: str = Query(default=None)):
    if not room:
        room = await generate_room_name()

    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity(name)
        .with_name(name)
        .with_grants(api.VideoGrants(room_join=True, room=room))
    )
    print(token)
    return {"token": token.to_jwt(), "room": room}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


def get_current_doctor(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]  # doctor email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/patients")
def read_patients(doctor_email: str = Depends(get_current_doctor)):
    # logic to show patient data
    return {"message": f"Data for doctor {doctor_email}"}


@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    doctor = db.query(Doctor).filter(Doctor.email == form_data.username).first()
    if not doctor or not verify_password(form_data.password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": doctor.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "doctor": {"id": doctor.id, "name": doctor.name, "email": doctor.email},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
