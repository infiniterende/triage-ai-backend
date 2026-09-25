"""
Tool implementation shared by the voice agent (see agent.py, where it is
exposed to the model as a `function_tool`).

`save_patient_assessment` stores the patient record, runs the clinical
pathway engine on the structured answers and returns the disposition so the
agent can read it back to the patient.
"""

from __future__ import annotations

import logging
from typing import Optional

from db import SessionLocal
from models import Patient
from pathways import ClinicalFindings, run_pathway

logger = logging.getLogger("agilance-voice.tools")


class PatientTools:
    def __init__(self, room_name: str = "", participant_name: str = "") -> None:
        self.room_name = room_name
        self.participant_name = participant_name
        self.patient_id: Optional[int] = None
        self.last_result = None

    def save_patient_assessment(
        self,
        *,
        name: str,
        age: int,
        sex: str,
        phone_number: str = "",
        pain_quality: str = "unknown",
        substernal: bool = False,
        radiates_to_arm_or_jaw: bool = False,
        exertional: bool = False,
        relieved_by_rest: bool = False,
        ongoing: bool = False,
        shortness_of_breath: bool = False,
        sweating: bool = False,
        nausea: bool = False,
        hypertension: bool = False,
        diabetes: bool = False,
        hyperlipidemia: bool = False,
        smoking: bool = False,
        heart_disease: bool = False,
    ) -> str:
        findings = ClinicalFindings.from_dict(
            {
                "age": age,
                "sex": sex,
                "chest_pain": {
                    "present": True,
                    "quality": pain_quality,
                    "location": "substernal" if substernal else None,
                    "radiation": ["arm", "jaw"] if radiates_to_arm_or_jaw else [],
                    "exertional": exertional,
                    "relieved_by_rest": relieved_by_rest,
                    "ongoing": ongoing,
                },
                "symptoms": {"dyspnea": shortness_of_breath, "diaphoresis": sweating, "nausea": nausea},
                "history": {
                    "hypertension": hypertension,
                    "diabetes": diabetes,
                    "hyperlipidemia": hyperlipidemia,
                    "smoking": smoking,
                    "coronary_artery_disease": heart_disease,
                },
            }
        )
        result = run_pathway(findings)
        self.last_result = result
        yn = lambda v: "Yes" if v else "No"  # noqa: E731

        with SessionLocal() as db:
            patient = Patient(
                name=name or self.participant_name,
                gender=sex,
                age=age,
                phone_number=phone_number or None,
                pain_quality=pain_quality,
                location=yn(substernal),
                stress=yn(exertional),
                sob=yn(shortness_of_breath),
                hypertension=yn(hypertension),
                diabetes=yn(diabetes),
                hyperlipidemia=yn(hyperlipidemia),
                smoking=yn(smoking),
                probability=int(result.risk_percent or 0),
            )
            db.add(patient)
            db.commit()
            db.refresh(patient)
            self.patient_id = patient.id

        logger.info(
            "saved patient %s (%s) risk %s%% → %s",
            patient.id, name, result.risk_percent, result.disposition.level.value,
        )
        primary = result.primary
        return (
            f"Saved. Risk level: {result.risk_percent}% ({result.disposition.level.value}). "
            f"Leading concern: {primary.name if primary else 'none'}. "
            f"Tell the patient: {result.disposition.headline}. {result.disposition.patient_message}"
        )
