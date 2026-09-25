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
from auth import verify_password, create_access_token

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


class VoiceRequest(BaseModel):
    session_id: str
    responses: List[Dict[str, Any]]


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


class ChestPainTriageSystem:
    def __init__(self):
        self.risk_factors = {
            "crushing": 3,
            "pressure": 3,
            "elephant": 3,
            "radiating": 2,
            "shortness": 2,
            "sweating": 2,
            "diaphoresis": 2,
            "nausea": 1,
            "heart_disease": 3,
            "diabetes": 2,
            "smoking": 1,
            "severe_pain": 2,
            "worst_pain": 3,
            "dying": 3,
            "arm_pain": 2,
            "jaw_pain": 2,
            "back_pain": 1,
        }

        self.questions = [
            "Can you describe your chest pain? Is it sharp, crushing, burning, or pressure-like?",
            "When did the pain start? Is it constant or does it come and go?",
            "On a scale of 1 to 10, how severe is your pain?",
            "Do you have any shortness of breath, nausea, sweating, or pain radiating to your arm, jaw, or back?",
            "Do you have any history of heart disease, diabetes, high blood pressure, or smoking?",
            "Are you currently taking any medications, especially heart medications?",
            "Have you had similar episodes before? If so, what happened?",
            "Are you experiencing any dizziness, lightheadedness, or feeling faint?",
        ]

        self.emergency_keywords = [
            r"can\'?t breathe|cannot breathe|difficulty breathing",
            r"worst pain|never felt pain like this|most severe",
            r"think I\'?m dying|feel like I\'?m dying|going to die",
            r"crushing|elephant on chest|heavy weight",
            r"heart attack|having a heart attack",
            r"chest tightness with sweating",
            r"pain down.*arm|arm pain with chest|jaw pain with chest",
        ]

        self.medical_patterns = {
            "crushing_pain": r"crushing|elephant|heavy pressure|weight on chest|vice",
            "pressure_pain": r"pressure|tight|squeezing|band around chest|constricting",
            "radiating_pain": r"radiating|spreading|arm pain|jaw pain|back pain|left arm|shoulder pain",
            "associated_symptoms": r"short of breath|can\'?t breathe|breathing difficulty|dyspnea",
            "autonomic_symptoms": r"sweating|diaphoresis|clammy|cold sweat|nausea|vomiting",
            "cardiac_history": r"heart disease|cardiac|coronary|heart attack|myocardial|angina|stent|bypass",
            "risk_factors": r"diabetes|diabetic|smoking|high blood pressure|hypertension",
            "severity_high": r"10.*10|worst pain|unbearable|excruciating|severe",
            "duration": r"(\d+)\s*(minute|hour|day)s?\s*ago|started\s*(\d+)",
        }

    def analyze_transcript(
        self, transcript: str, conversation_context: Dict = None
    ) -> Dict:
        """Analyze the patient's transcript for medical risk factors"""
        transcript_lower = transcript.lower()
        risk_score = 0
        detected_factors = []

        # Check for emergency keywords first
        for pattern in self.emergency_keywords:
            if re.search(pattern, transcript_lower):
                return {
                    "risk_score": 10,
                    "risk_level": "emergency",
                    "is_emergency": True,
                    "detected_factors": ["emergency_keywords"],
                    "recommendation": "EMERGENCY: Call 911 immediately or go to the nearest emergency room. Do not drive yourself.",
                }

        # Analyze medical patterns
        for factor, pattern in self.medical_patterns.items():
            if re.search(pattern, transcript_lower):
                if factor == "crushing_pain":
                    risk_score += self.risk_factors["crushing"]
                    detected_factors.append("crushing chest pain")
                elif factor == "pressure_pain":
                    risk_score += self.risk_factors["pressure"]
                    detected_factors.append("pressure-type chest pain")
                elif factor == "radiating_pain":
                    risk_score += self.risk_factors["radiating"]
                    detected_factors.append("radiating pain")
                elif factor == "associated_symptoms":
                    risk_score += self.risk_factors["shortness"]
                    detected_factors.append("breathing difficulty")
                elif factor == "autonomic_symptoms":
                    risk_score += self.risk_factors["sweating"]
                    detected_factors.append("associated symptoms")
                elif factor == "cardiac_history":
                    risk_score += self.risk_factors["heart_disease"]
                    detected_factors.append("cardiac history")
                elif factor == "risk_factors":
                    risk_score += self.risk_factors["diabetes"]
                    detected_factors.append("cardiovascular risk factors")
                elif factor == "severity_high":
                    risk_score += self.risk_factors["severe_pain"]
                    detected_factors.append("severe pain")

        # Extract pain severity score if mentioned
        severity_match = re.search(r"(\d+)\s*(?:out of|/|\s)\s*10", transcript_lower)
        if severity_match:
            severity = int(severity_match.group(1))
            if severity >= 8:
                risk_score += 3
                detected_factors.append(f"severe pain ({severity}/10)")
            elif severity >= 6:
                risk_score += 2
                detected_factors.append(f"moderate-severe pain ({severity}/10)")

        # Determine risk level
        if risk_score >= 6:
            risk_level = "emergency"
            is_emergency = True
        elif risk_score >= 4:
            risk_level = "high"
            is_emergency = False
        elif risk_score >= 2:
            risk_level = "medium"
            is_emergency = False
        else:
            risk_level = "low"
            is_emergency = False

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_emergency": is_emergency,
            "detected_factors": detected_factors,
        }

    def get_recommendation(self, analysis: Dict, question_count: int) -> str:
        """Generate appropriate medical recommendation based on risk analysis"""
        risk_level = analysis["risk_level"]

        if risk_level == "emergency":
            return "Based on your symptoms, this appears to be a medical emergency. Call 911 immediately or have someone drive you to the nearest emergency room. Do not drive yourself. Time is critical for heart attacks."

        elif risk_level == "high":
            return "Your symptoms are concerning and suggest you should seek immediate medical attention. Please go to an emergency room or call 911 if symptoms worsen. Do not wait - chest pain with these characteristics needs urgent evaluation."

        elif risk_level == "medium":
            return "Your symptoms warrant prompt medical evaluation. Please contact your doctor immediately or visit an urgent care center within the next 2-4 hours. If symptoms worsen or you develop new symptoms, go to the emergency room."

        else:  # low risk
            if question_count < 3:
                return "Thank you for that information. While your symptoms may be lower risk, chest pain should always be evaluated by a healthcare professional. Let me ask a few more questions to better assess your situation."
            else:
                return "Based on our discussion, your symptoms appear to be lower risk, but chest pain should still be evaluated by a healthcare professional. Please schedule an appointment with your primary care doctor within the next day or two. If symptoms worsen, seek immediate care."

    def get_next_question(self, question_count: int, analysis: Dict) -> Optional[str]:
        """Get the next appropriate question based on current assessment"""
        if analysis["is_emergency"] or question_count >= len(self.questions):
            return None

        return self.questions[question_count]


# Initialize triage system
triage_system = ChestPainTriageSystem()


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )

        transcript = resp.strip()

        os.remove(tmp_path)
        print(transcript)
        return {"transcript": transcript}

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/triage")
async def process_triage(
    audio: UploadFile = File(...), conversation_context: str = Form("{}")
):
    try:
        # Parse conversation context
        context = json.loads(conversation_context)

        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_filename = temp_file.name

        try:
            with open(tmp_path, "rb") as audio_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    temperature=0.0,
                )
            transcript = (
                result.strip()
                if isinstance(result, str)
                else result.get("text", "").strip()
            )
            if not transcript:
                return JSONResponse({"error": "No speech detected"}, status_code=400)

            # Append user message to conversation history
            context.setdefault("messages", []).append(
                {"sender": "user", "text": transcript}
            )

            # Analyze transcript if needed
            analysis = triage_system.analyze_transcript(transcript, context)

            # Count agent questions asked
            question_count = len(
                [m for m in context["messages"] if m["sender"] == "agent"]
            )

            recommendation = triage_system.get_recommendation(analysis, question_count)

            # Build ChatGPT messages from conversation history for context
            chat_messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
            ]
            for m in context["messages"]:
                role = "assistant" if m["sender"] == "agent" else "user"
                chat_messages.append({"role": role, "content": m["text"]})

            # Get ChatGPT response for next agent message (question or reply)
            chat_response = client.chat.completions.create(
                model="gpt-4",
                messages=chat_messages,
                temperature=0.7,
            )
            assistant_reply = chat_response.choices[0].message.content.strip()

            # Append agent reply to conversation history
            context["messages"].append({"sender": "agent", "text": assistant_reply})

            response = {
                "transcript": transcript,
                "assistant_reply": assistant_reply,
                "risk_score": analysis["risk_score"],
                "risk_level": analysis["risk_level"],
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "conversation_context": context,  # return updated conversation for frontend
            }

            # Delay 5 seconds before returning to simulate pause (optional)
            await asyncio.sleep(5)

        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    except Exception as e:
        logger.error(f"Error processing triage request: {str(e)}")
        return JSONResponse(
            content={"error": f"Processing failed: {str(e)}"}, status_code=500
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "model": "whisper-base",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/questions", methods=["GET"])
def get_questions():
    """Get all triage questions"""
    return jsonify(
        {
            "questions": triage_system.questions,
            "total_questions": len(triage_system.questions),
        }
    )


@app.post("/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Read audio file
        audio_data = await audio.read()

        # Use OpenAI Whisper for transcription
        response = client.Audio.transcribe(
            model="whisper-1", file=io.BytesIO(audio_data), response_format="text"
        )

        return {"transcription": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/generate-speech")
async def generate_speech(text: str):
    """Generate speech from text using OpenAI TTS"""
    try:
        response = client.Audio.speech.create(model="tts-1", voice="nova", input=text)

        return {"audio_url": "data:audio/mp3;base64," + response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/chat")
async def voice_chat(request: VoiceRequest):
    """Handle voice conversation with ChatGPT"""
    try:
        # Create conversation context
        messages = [
            {
                "role": "system",
                "content": """You are a medical AI assistant conducting a chest pain assessment. 
                Ask one question at a time from this list, in order:
                1. What is your age?
                2. Are you male or female?
                3. How would you describe your chest pain? Is it crushing, stabbing, burning, or aching?
                4. On a scale of 1-10, how severe is your chest pain?
                5. When did the chest pain start?
                6. Does the pain radiate to your arm, jaw, neck, or back?
                7. Are you experiencing shortness of breath?
                8. Are you feeling nauseous or have you vomited?
                9. Are you sweating more than usual?
                10. Do you have a history of heart disease?
                11. Do you have diabetes?
                12. Do you smoke or have you smoked in the past?
                13. Do you have high blood pressure?
                
                Keep questions brief and clear. Wait for the user's response before asking the next question.
                """,
            }
        ]

        # Add previous conversation
        for response in request.responses:
            if response.get("role") == "assistant":
                messages.append({"role": "assistant", "content": response["content"]})
            elif response.get("role") == "user":
                messages.append({"role": "user", "content": response["content"]})

        # Get ChatGPT response
        chat_response = client.ChatCompletion.create(
            model="gpt-4", messages=messages, max_tokens=150, temperature=0.7
        )

        response_text = chat_response.choices[0].message.content

        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess")
async def assess_risk(request: AssessmentRequest):
    """Assess heart attack risk based on responses"""
    try:
        assessment = calculate_heart_attack_risk(request.responses)

        # Store session data
        sessions[request.session_id] = {
            "responses": [r.dict() for r in request.responses],
            "assessment": assessment,
            "timestamp": datetime.now().isoformat(),
        }

        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
ALGORITHM = os.getenv("ALGORITHM")


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
    return {"access_token": token, doctor: doctor, "token_type": "bearer"}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
