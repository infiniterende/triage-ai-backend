import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Text,
    TIMESTAMP,
    func,
    text,
    JSON,
    ForeignKey,
    DateTime,
)
from sqlalchemy import Enum as SQLEnum
from enum import Enum

import pandas as pd

from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from dotenv import load_dotenv

load_dotenv()
# Read the database URL from environment variable
DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")
# Create engine
print(DATABASE_URL)
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()
# Base class for declarative models
Base = declarative_base()


class SpeakerType(str, Enum):
    patient = "patient"
    agent = "agent"


class VoiceSessionStatus(str, Enum):
    active = "active"
    completed = "completed"
    aborted = "aborted"


# Define your model class (example for patients table)
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    name = Column(String(100))
    gender = Column(String(100))
    age = Column(Integer, nullable=False)
    phone_number = Column(String(100))
    pain_quality = Column(Text, nullable=False)
    location = Column(Text, nullable=False)
    stress = Column(Text, nullable=False)
    sob = Column(Text, nullable=False)  # shortness of breath
    hypertension = Column(Text, nullable=False)
    diabetes = Column(Text, nullable=False)
    hyperlipidemia = Column(Text, nullable=False)
    smoking = Column(Text, nullable=False)
    probability = Column(Integer, nullable=False)
    session = relationship("VoiceSession", back_populates="patient")


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    messages = relationship(
        "Message", back_populates="session", cascade="all, delete-orphan"
    )
    conversation_history = relationship(
        "Message",
        overlaps="messages",  # <--- add this
        viewonly=True,  # optional if you don't intend to write through this relationship
    )
    current_question = Column(Integer, default=0)
    responses = Column(JSON, default={})
    assessment_complete = Column(Boolean, default=False)
    risk_score = Column(Integer, default=0)


class VoiceSession(Base):
    __tablename__ = "voice_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)

    transcript = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow())
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    session_status = Column(
        SQLEnum(VoiceSessionStatus, name="voice_session_status", default="active")
    )
    risk_score = Column(Integer, nullable=True)
    assessment_complete = Column(Boolean, default=False)
    transcripts = relationship(
        "VoiceTranscript", back_populates="session", cascade="all, delete-orphan"
    )
    patient = relationship("Patient", back_populates="session")
    audio_chunk = relationship("AudioChunk", back_populates="session")


class VoiceTranscript(Base):
    __tablename__ = "voice_transcripts"
    id = Column(Integer, primary_key=True)
    voice_session_id = Column(Integer, ForeignKey("voice_sessions.id"), index=True)
    speaker = Column((SQLEnum(SpeakerType, name="speaker_type", native_enum=True)))
    text = Column(Text, nullable=False)
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    session = relationship("VoiceSession", back_populates="transcripts")


class AudioChunk(Base):
    __tablename__ = "audio_chunks"
    id = Column(Integer, primary_key=True)
    voice_session_id = Column(ForeignKey("voice_sessions.id"))
    file_path = Column(String)
    start_ms = Column(Integer)
    end_ms = Column(Integer)
    session = relationship("VoiceSession", back_populates="audio_chunk")
    created_at = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Clinical pathway + care coordination tables (new tables only, so
# `Base.metadata.create_all` adds them without touching existing schemas).
# ---------------------------------------------------------------------------


class PathwayEvaluation(Base):
    """One run of the clinical pathway engine, linked to the conversation that
    produced it and (once known) the patient record."""

    __tablename__ = "pathway_evaluations"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)  # ChatSession.session_id or LiveKit room
    source = Column(String, default="text")  # "text" | "voice"
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True, index=True)
    primary_pathway = Column(String, nullable=True)
    disposition = Column(String, nullable=True)  # emergency | urgent | prompt | routine
    risk_percent = Column(Integer, nullable=True)
    result = Column(JSON, nullable=False)  # full PathwayResult.to_dict()
    created_at = Column(DateTime, default=datetime.utcnow)


class AppointmentStatus(str, Enum):
    requested = "requested"
    scheduled = "scheduled"
    completed = "completed"
    cancelled = "cancelled"


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    scheduled_for = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=30)
    appointment_type = Column(String, default="follow_up")  # follow_up | urgent | new_patient | telehealth
    status = Column(String, default=AppointmentStatus.requested.value)
    reason = Column(Text, nullable=True)
    location = Column(String, nullable=True)
    pathway_evaluation_id = Column(Integer, ForeignKey("pathway_evaluations.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DoctorNote(Base):
    __tablename__ = "doctor_notes"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    author_name = Column(String, nullable=True)
    note_type = Column(String, default="progress")  # progress | plan | follow_up | message_to_patient
    content = Column(Text, nullable=False)
    visible_to_patient = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)




# Drop all tables
# Base.metadata.drop_all(bind=engine)

# with engine.begin() as conn:
#     conn.execute(text("SET statement_timeout = 0;"))  # disable timeout
#     conn.execute(text("DROP TABLE IF EXISTS patients CASCADE;"))
#     conn.execute(text("DROP TABLE IF EXISTS doctors CASCADE;"))
#     conn.execute(text("DROP TABLE IF EXISTS chat_sessions CASCADE;"))
#     conn.execute(text("DROP TABLE IF EXISTS messages CASCADE;"))
#     conn.execute(text("DROP TABLE IF EXISTS conversations CASCADE;"))
# Recreate all tables
# Base.metadata.drop_all(engine)
# Base.metadata.create_all(bind=engine, checkfirst=True)
print("All tables created.")

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

# seed()


# Create tables in the database
def init_db():
    Base.metadata.create_all(bind=engine, checkfirst=True)

    patients = session.query(Patient).all()
    doctors = session.query(Doctor).all()
    sessions = session.query(ChatSession).all()
    for patient in patients:
        print(
            f"{patient.id}: {patient.name}, {patient.age}, {patient.gender}, "
            f"{patient.phone_number}, {patient.pain_quality}, {patient.location}, "
            f"Stress: {patient.stress}, SOB: {patient.sob}, HTN: {patient.hypertension}, "
            f"DM: {patient.diabetes}, HLD: {patient.hyperlipidemia}, Smoke: {patient.smoking}, "
            f"Prob: {patient.probability}"
        )


# if __name__ == "__main__":
# init_db()
# print("Tables created!")
