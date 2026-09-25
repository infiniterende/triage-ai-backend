from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from dotenv import load_dotenv
import os
from models import Patient

from db_url import get_database_url

load_dotenv()
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
DATABASE_URL = get_database_url()


class DatabaseDriver:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_patient(
        self,
        name: str,
        gender: str,
        age: int,
        phone_number: str,
        pain_quality: str,
        location: str,
        stress: bool,
        sob: bool,
        hypertension: bool,
        diabetes: bool,
        hyperlipidemia: bool,
        smoking: bool,
        probability: int,
    ) -> Patient:
        with self.SessionLocal() as session:
            patient = Patient(
                name=name,
                gender=gender,
                age=age,
                phone_number=phone_number,
                pain_quality=pain_quality,
                location=location,
                stress=stress,
                sob=sob,
                hypertension=hypertension,
                diabetes=diabetes,
                hyperlipidemia=hyperlipidemia,
                smoking=smoking,
                probability=probability,
            )
            session.add(patient)
            session.commit()
            session.refresh(patient)
            return patient

    def get_patients(self) -> list[Patient]:
        with self.SessionLocal() as session:
            return session.query(Patient).all()

    def get_patient_by_id(self, patient_id: int) -> Patient | None:
        with self.SessionLocal() as session:
            return session.query(Patient).filter(Patient.id == patient_id).first()
