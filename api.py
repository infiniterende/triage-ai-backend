"""
Function tools available to the voice agent (LiveKit ``FunctionContext``).

The Realtime model calls ``save_patient_assessment`` once it has collected the
patient's details; the tool stores the record, runs the clinical pathway
engine on the structured answers and returns the disposition so the agent can
read it back to the patient.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

from livekit.agents import llm

from db import SessionLocal
from models import Patient
from pathways import ClinicalFindings, run_pathway

logger = logging.getLogger("agilance-voice.tools")


class AssistantFnc(llm.FunctionContext):
    def __init__(self, room_name: str = "", participant_name: str = "") -> None:
        super().__init__()
        self.room_name = room_name
        self.participant_name = participant_name
        self.patient_id: Optional[int] = None
        self.last_result = None

    @llm.ai_callable(
        description=(
            "Save the patient's chest pain assessment once you have their name, age, "
            "sex, pain description, triggers and history, then get the risk level and "
            "recommendation to read back to them."
        )
    )
    def save_patient_assessment(
        self,
        name: Annotated[str, llm.TypeInfo(description="Patient's name")],
        age: Annotated[int, llm.TypeInfo(description="Age in years")],
        sex: Annotated[str, llm.TypeInfo(description="'male' or 'female'")],
        phone_number: Annotated[str, llm.TypeInfo(description="Phone number, or empty string")],
        pain_quality: Annotated[str, llm.TypeInfo(description="pressure, sharp, burning, tearing or dull")],
        substernal: Annotated[bool, llm.TypeInfo(description="Pain in the centre of the chest / behind the breastbone")],
        radiates_to_arm_or_jaw: Annotated[bool, llm.TypeInfo(description="Pain spreads to arm, jaw or neck")],
        exertional: Annotated[bool, llm.TypeInfo(description="Brought on by physical activity or stress")],
        relieved_by_rest: Annotated[bool, llm.TypeInfo(description="Eases within minutes of resting")],
        ongoing: Annotated[bool, llm.TypeInfo(description="Pain is happening right now")],
        shortness_of_breath: Annotated[bool, llm.TypeInfo(description="Short of breath")],
        sweating: Annotated[bool, llm.TypeInfo(description="Sweating or clammy with the pain")],
        nausea: Annotated[bool, llm.TypeInfo(description="Nausea or vomiting")],
        hypertension: Annotated[bool, llm.TypeInfo(description="History of high blood pressure")],
        diabetes: Annotated[bool, llm.TypeInfo(description="History of diabetes")],
        hyperlipidemia: Annotated[bool, llm.TypeInfo(description="History of high cholesterol")],
        smoking: Annotated[bool, llm.TypeInfo(description="Current or past smoker")],
        heart_disease: Annotated[bool, llm.TypeInfo(description="Known heart disease, prior heart attack, stent or bypass")],
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
