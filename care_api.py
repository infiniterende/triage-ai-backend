"""
Care-coordination API used by the patient and clinician dashboards.

Mount with ``app.include_router(care_router)``. Everything here is read/write
against the existing SQLAlchemy models plus the new pathway / appointment /
note tables in ``models.py``.

Patient-facing
    GET  /api/patients/{id}/summary        one call that feeds the patient dashboard
    GET  /api/patients/{id}/conversations  logged AI conversations (text + voice)
    GET  /api/chat/{session_id}/messages   full transcript of one conversation
    GET  /api/patients/{id}/pathway        latest pathway evaluation
    GET  /api/patients/{id}/appointments
    POST /api/appointments                 schedule / request an appointment
    PATCH /api/appointments/{id}           reschedule, confirm, cancel
    GET  /api/patients/{id}/notes          doctor's notes visible to the patient

Clinician-facing
    GET  /api/clinician/patients           patients enriched with latest pathway result
    GET  /api/clinician/patients/{id}      full chart: findings, pathway, transcripts, notes, appointments
    POST /api/patients/{id}/notes          add a note
    GET  /api/doctors                      for the scheduling picker
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import (
    Appointment,
    AppointmentStatus,
    ChatSession,
    Doctor,
    DoctorNote,
    Message,
    PathwayEvaluation,
    Patient,
)

router = APIRouter(prefix="/api", tags=["care"])


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #

def _patient_dict(p: Patient) -> Dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "gender": p.gender,
        "age": p.age,
        "phone_number": p.phone_number,
        "pain_quality": p.pain_quality,
        "location": p.location,
        "stress": p.stress,
        "sob": p.sob,
        "hypertension": p.hypertension,
        "diabetes": p.diabetes,
        "hyperlipidemia": p.hyperlipidemia,
        "smoking": p.smoking,
        "probability": p.probability,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _evaluation_dict(e: PathwayEvaluation) -> Dict[str, Any]:
    return {
        "id": e.id,
        "session_id": e.session_id,
        "source": e.source,
        "patient_id": e.patient_id,
        "primary_pathway": e.primary_pathway,
        "disposition": e.disposition,
        "risk_percent": e.risk_percent,
        "result": e.result,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


def _appointment_dict(a: Appointment, doctor: Optional[Doctor] = None) -> Dict[str, Any]:
    return {
        "id": a.id,
        "patient_id": a.patient_id,
        "doctor_id": a.doctor_id,
        "doctor_name": doctor.name if doctor else None,
        "scheduled_for": a.scheduled_for.isoformat() if a.scheduled_for else None,
        "duration_minutes": a.duration_minutes,
        "appointment_type": a.appointment_type,
        "status": a.status,
        "reason": a.reason,
        "location": a.location,
        "pathway_evaluation_id": a.pathway_evaluation_id,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def _note_dict(n: DoctorNote) -> Dict[str, Any]:
    return {
        "id": n.id,
        "patient_id": n.patient_id,
        "doctor_id": n.doctor_id,
        "author_name": n.author_name,
        "note_type": n.note_type,
        "content": n.content,
        "visible_to_patient": n.visible_to_patient,
        "created_at": n.created_at.isoformat() if n.created_at else None,
    }


def _message_dict(m: Message) -> Dict[str, Any]:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def _conversation_summaries(db: Session, patient_id: int) -> List[Dict[str, Any]]:
    """Conversations are linked to a patient through their pathway evaluations."""
    evals = (
        db.query(PathwayEvaluation)
        .filter(PathwayEvaluation.patient_id == patient_id)
        .order_by(PathwayEvaluation.created_at.desc())
        .all()
    )
    out: List[Dict[str, Any]] = []
    seen = set()
    for e in evals:
        if e.session_id in seen:
            continue
        seen.add(e.session_id)
        session = db.query(ChatSession).filter(ChatSession.session_id == e.session_id).first()
        messages = sorted(session.messages, key=lambda m: m.id) if session else []
        result = e.result or {}
        disposition = (result.get("disposition") or {}).get("level") or e.disposition
        primary = (result.get("primary_pathway") or {}).get("name") or e.primary_pathway
        out.append(
            {
                "session_id": e.session_id,
                "source": e.source,
                "started_at": messages[0].created_at.isoformat() if messages and messages[0].created_at else (e.created_at.isoformat() if e.created_at else None),
                "message_count": len(messages),
                "preview": next((m.content for m in messages if m.role == "user"), None),
                "risk_percent": e.risk_percent,
                "disposition": disposition,
                "primary_pathway": primary,
                "evaluation_id": e.id,
                "assessment_complete": bool(session.assessment_complete) if session else True,
            }
        )
    return out


def _get_patient_or_404(db: Session, patient_id: int) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


# --------------------------------------------------------------------------- #
# Patient-facing
# --------------------------------------------------------------------------- #

@router.get("/patients/{patient_id}/summary")
def patient_summary(patient_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    patient = _get_patient_or_404(db, patient_id)
    latest = (
        db.query(PathwayEvaluation)
        .filter(PathwayEvaluation.patient_id == patient_id)
        .order_by(PathwayEvaluation.created_at.desc())
        .first()
    )
    appointments = (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient_id)
        .order_by(Appointment.scheduled_for.asc())
        .all()
    )
    doctors = {d.id: d for d in db.query(Doctor).all()}
    notes = (
        db.query(DoctorNote)
        .filter(DoctorNote.patient_id == patient_id, DoctorNote.visible_to_patient == True)  # noqa: E712
        .order_by(DoctorNote.created_at.desc())
        .all()
    )
    return {
        "patient": _patient_dict(patient),
        "latest_evaluation": _evaluation_dict(latest) if latest else None,
        "conversations": _conversation_summaries(db, patient_id),
        "appointments": [_appointment_dict(a, doctors.get(a.doctor_id)) for a in appointments],
        "notes": [_note_dict(n) for n in notes],
    }


@router.get("/patients/{patient_id}/conversations")
def patient_conversations(patient_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    _get_patient_or_404(db, patient_id)
    return _conversation_summaries(db, patient_id)


@router.get("/chat/{session_id}/messages")
def chat_messages(session_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    evaluation = (
        db.query(PathwayEvaluation)
        .filter(PathwayEvaluation.session_id == session_id)
        .order_by(PathwayEvaluation.created_at.desc())
        .first()
    )
    return {
        "session_id": session_id,
        "assessment_complete": session.assessment_complete,
        "messages": [_message_dict(m) for m in sorted(session.messages, key=lambda m: m.id)],
        "pathway": evaluation.result if evaluation else None,
    }


@router.get("/patients/{patient_id}/pathway")
def patient_pathway(patient_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    _get_patient_or_404(db, patient_id)
    latest = (
        db.query(PathwayEvaluation)
        .filter(PathwayEvaluation.patient_id == patient_id)
        .order_by(PathwayEvaluation.created_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No pathway evaluation for this patient yet")
    return _evaluation_dict(latest)


@router.get("/patients/{patient_id}/appointments")
def patient_appointments(patient_id: int, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    _get_patient_or_404(db, patient_id)
    doctors = {d.id: d for d in db.query(Doctor).all()}
    rows = (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient_id)
        .order_by(Appointment.scheduled_for.asc())
        .all()
    )
    return [_appointment_dict(a, doctors.get(a.doctor_id)) for a in rows]


class AppointmentCreate(BaseModel):
    patient_id: int
    doctor_id: Optional[int] = None
    scheduled_for: datetime
    duration_minutes: int = 30
    appointment_type: str = "follow_up"
    reason: Optional[str] = None
    location: Optional[str] = None
    pathway_evaluation_id: Optional[int] = None
    status: str = AppointmentStatus.requested.value


@router.post("/appointments", status_code=201)
def create_appointment(body: AppointmentCreate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    _get_patient_or_404(db, body.patient_id)
    doctor = db.query(Doctor).filter(Doctor.id == body.doctor_id).first() if body.doctor_id else None
    if body.doctor_id and not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    if body.status not in {s.value for s in AppointmentStatus}:
        raise HTTPException(status_code=422, detail="Invalid status")
    appt = Appointment(**body.dict())
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return _appointment_dict(appt, doctor)


class AppointmentUpdate(BaseModel):
    scheduled_for: Optional[datetime] = None
    doctor_id: Optional[int] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    location: Optional[str] = None
    appointment_type: Optional[str] = None


@router.patch("/appointments/{appointment_id}")
def update_appointment(appointment_id: int, body: AppointmentUpdate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if "status" in updates and updates["status"] not in {s.value for s in AppointmentStatus}:
        raise HTTPException(status_code=422, detail="Invalid status")
    for k, v in updates.items():
        setattr(appt, k, v)
    db.commit()
    db.refresh(appt)
    doctor = db.query(Doctor).filter(Doctor.id == appt.doctor_id).first() if appt.doctor_id else None
    return _appointment_dict(appt, doctor)


@router.get("/patients/{patient_id}/notes")
def patient_notes(patient_id: int, include_private: bool = False, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    _get_patient_or_404(db, patient_id)
    q = db.query(DoctorNote).filter(DoctorNote.patient_id == patient_id)
    if not include_private:
        q = q.filter(DoctorNote.visible_to_patient == True)  # noqa: E712
    return [_note_dict(n) for n in q.order_by(DoctorNote.created_at.desc()).all()]


class NoteCreate(BaseModel):
    content: str
    note_type: str = "progress"
    author_name: Optional[str] = None
    doctor_id: Optional[int] = None
    visible_to_patient: bool = True


@router.post("/patients/{patient_id}/notes", status_code=201)
def add_note(patient_id: int, body: NoteCreate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    _get_patient_or_404(db, patient_id)
    if not body.content.strip():
        raise HTTPException(status_code=422, detail="Note content is required")
    note = DoctorNote(patient_id=patient_id, **body.dict())
    db.add(note)
    db.commit()
    db.refresh(note)
    return _note_dict(note)


@router.get("/doctors")
def list_doctors(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    return [{"id": d.id, "name": d.name, "email": d.email} for d in db.query(Doctor).order_by(Doctor.name).all()]


# --------------------------------------------------------------------------- #
# Clinician-facing
# --------------------------------------------------------------------------- #

@router.get("/clinician/patients")
def clinician_patients(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Patients enriched with their latest pathway evaluation for the queue."""
    patients = db.query(Patient).order_by(Patient.created_at.desc().nullslast()).all()
    latest_by_patient: Dict[int, PathwayEvaluation] = {}
    for e in db.query(PathwayEvaluation).order_by(PathwayEvaluation.created_at.desc()).all():
        if e.patient_id is not None and e.patient_id not in latest_by_patient:
            latest_by_patient[e.patient_id] = e
    out = []
    for p in patients:
        row = _patient_dict(p)
        e = latest_by_patient.get(p.id)
        if e:
            result = e.result or {}
            row["pathway"] = {
                "evaluation_id": e.id,
                "primary_pathway": e.primary_pathway,
                "primary_pathway_name": (result.get("primary_pathway") or {}).get("name"),
                "likelihood": (result.get("primary_pathway") or {}).get("likelihood"),
                "disposition": e.disposition,
                "risk_percent": e.risk_percent,
                "red_flags": [r.get("label") for r in result.get("red_flags", [])],
                "evaluated_at": e.created_at.isoformat() if e.created_at else None,
            }
        else:
            row["pathway"] = None
        out.append(row)
    return out


@router.get("/clinician/patients/{patient_id}")
def clinician_patient_chart(patient_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    patient = _get_patient_or_404(db, patient_id)
    evaluations = (
        db.query(PathwayEvaluation)
        .filter(PathwayEvaluation.patient_id == patient_id)
        .order_by(PathwayEvaluation.created_at.desc())
        .all()
    )
    conversations = _conversation_summaries(db, patient_id)
    transcripts = []
    for c in conversations:
        session = db.query(ChatSession).filter(ChatSession.session_id == c["session_id"]).first()
        if session:
            transcripts.append(
                {
                    "session_id": c["session_id"],
                    "messages": [_message_dict(m) for m in sorted(session.messages, key=lambda m: m.id)],
                }
            )
    doctors = {d.id: d for d in db.query(Doctor).all()}
    appointments = (
        db.query(Appointment)
        .filter(Appointment.patient_id == patient_id)
        .order_by(Appointment.scheduled_for.asc())
        .all()
    )
    notes = (
        db.query(DoctorNote)
        .filter(DoctorNote.patient_id == patient_id)
        .order_by(DoctorNote.created_at.desc())
        .all()
    )
    return {
        "patient": _patient_dict(patient),
        "latest_evaluation": _evaluation_dict(evaluations[0]) if evaluations else None,
        "evaluations": [_evaluation_dict(e) for e in evaluations],
        "conversations": conversations,
        "transcripts": transcripts,
        "appointments": [_appointment_dict(a, doctors.get(a.doctor_id)) for a in appointments],
        "notes": [_note_dict(n) for n in notes],
    }
