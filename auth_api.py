"""
Clinician authentication used by the Next.js frontend (NextAuth credentials
provider and the /signup page). Passwords are bcrypt-hashed in the
``doctors`` table; the frontend keeps the session (JWT via NextAuth).

POST /auth/register            {name, email, password, specialty?} → 201 doctor
POST /auth/verify-credentials  {email, password}                  → 200 doctor | 401
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import Doctor

router = APIRouter(prefix="/auth", tags=["auth"])


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def _doctor_dict(d: Doctor) -> Dict[str, Any]:
    return {"id": str(d.id), "name": d.name, "email": d.email}


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    specialty: Optional[str] = None  # accepted for forward compatibility; not stored yet


class CredentialsRequest(BaseModel):
    email: str
    password: str


@router.post("/register", status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    email = body.email.strip().lower()
    if not body.name.strip() or not email or len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Name, email and a password of at least 8 characters are required")
    if db.query(Doctor).filter(Doctor.email == email).first():
        raise HTTPException(status_code=409, detail="An account with that email already exists")
    doctor = Doctor(name=body.name.strip(), email=email, hashed_password=hash_password(body.password))
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return _doctor_dict(doctor)


@router.post("/verify-credentials")
def verify_credentials(body: CredentialsRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    doctor = db.query(Doctor).filter(Doctor.email == body.email.strip().lower()).first()
    if not doctor or not verify_password(body.password, doctor.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _doctor_dict(doctor)
