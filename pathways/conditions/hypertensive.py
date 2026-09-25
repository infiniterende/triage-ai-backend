"""
Hypertension pathway.

Acts as both an *acute* pathway (hypertensive emergency / urgency) and a
*chronic* pathway (blood-pressure staging per ACC/AHA 2017) so it appears in
the router ranking when there is an acute BP crisis and in the chronic risk
panel otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "hypertensive"
NAME = "Hypertensive emergency / urgency"
DESCRIPTION = (
    "Severely elevated blood pressure (≥180/120). An emergency when there is "
    "organ damage — headache, vision change, confusion, chest pain or breathlessness."
)


def stage_blood_pressure(sbp: Optional[int], dbp: Optional[int]) -> Optional[str]:
    """ACC/AHA 2017 categories. Returns None when no reading is available."""
    if sbp is None and dbp is None:
        return None
    sbp = sbp or 0
    dbp = dbp or 0
    if sbp >= 180 or dbp >= 120:
        return "crisis"
    if sbp >= 140 or dbp >= 90:
        return "stage_2"
    if sbp >= 130 or dbp >= 80:
        return "stage_1"
    if sbp >= 120 and dbp < 80:
        return "elevated"
    return "normal"


def assess(f: ClinicalFindings) -> ConditionAssessment:
    s, h, v = f.symptoms, f.history, f.vitals
    b = ScoreBuilder()
    stage = stage_blood_pressure(v.systolic_bp, v.diastolic_bp)

    end_organ = [
        label
        for cond, label in (
            (s.severe_headache, "Severe headache"),
            (s.vision_changes, "Vision changes"),
            (s.confusion, "Confusion"),
            (s.focal_neuro_deficit, "Focal weakness / speech difficulty"),
            (f.has_chest_pain, "Chest pain"),
            (s.dyspnea_at_rest, "Breathless at rest"),
        )
        if cond
    ]

    if stage == "crisis":
        b.add(True, 55, f"Blood pressure {v.systolic_bp}/{v.diastolic_bp} mmHg (≥180/120)")
        for label in end_organ:
            b.add(True, 12, f"{label} with severe hypertension")
    elif stage == "stage_2":
        b.add(True, 15, f"Blood pressure {v.systolic_bp}/{v.diastolic_bp} mmHg (stage 2)")
    elif stage is None:
        if h.hypertension or s.severe_headache:
            b.missing.append("Do you have a recent blood pressure reading? What was it?")

    b.add(h.hypertension, 8, "Known hypertension", "Have you been told you have high blood pressure?")
    b.add(h.chronic_kidney_disease, 4, "Chronic kidney disease")
    b.add(h.stimulant_or_cocaine_use, 6, "Stimulant or cocaine use")
    if h.hypertension and stage is None and s.severe_headache:
        b.add(True, 10, "Severe headache in a hypertensive patient")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    if stage == "crisis" and end_organ:
        disposition = Disposition.EMERGENCY
        summary = "Hypertensive emergency — severely raised BP with signs of organ involvement."
    elif stage == "crisis":
        disposition = Disposition.URGENT
        summary = "Hypertensive urgency — BP ≥180/120 without organ symptoms; needs same-day review."
    elif stage == "stage_2":
        disposition = Disposition.PROMPT
        summary = "Stage 2 hypertension — arrange clinician review within days."
    elif stage in ("stage_1", "elevated"):
        disposition = Disposition.ROUTINE
        summary = f"Blood pressure is {stage.replace('_', ' ')} — routine follow-up and lifestyle review."
    else:
        disposition = Disposition.ROUTINE
        summary = "No blood-pressure crisis identified." if stage is None else "Blood pressure in the normal range."

    extra: Dict[str, Any] = {"bp_stage": stage, "end_organ_symptoms": end_organ}
    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:2], summary, extra)


# --- Chronic view --------------------------------------------------------------

CHRONIC_ID = "hypertension_management"
CHRONIC_NAME = "Hypertension"


def assess_chronic(f: ClinicalFindings) -> Dict[str, Any]:
    v, h = f.vitals, f.history
    stage = stage_blood_pressure(v.systolic_bp, v.diastolic_bp)
    if stage is None and not h.hypertension:
        return {
            "pathway": CHRONIC_ID,
            "name": CHRONIC_NAME,
            "status": "not_applicable",
            "summary": "No hypertension history and no blood-pressure reading recorded.",
            "recommendations": ["Check blood pressure at least once a year."],
        }
    targets = "below 130/80 mmHg" if (h.diabetes or h.chronic_kidney_disease or f.known_cad()) else "below 130/80 mmHg (ACC/AHA) — discuss your personal target with your clinician"
    recs = [
        "Home blood-pressure monitoring: two readings, morning and evening, for 7 days before visits",
        "Limit sodium to under 2 g/day; DASH-style eating pattern",
        "150 minutes/week of moderate activity if cleared by your clinician",
        "Review adherence to any antihypertensive medication",
    ]
    if stage in ("stage_2", "crisis"):
        recs.insert(0, "Medication review — most people at this stage need treatment intensified")
    return {
        "pathway": CHRONIC_ID,
        "name": CHRONIC_NAME,
        "status": "active",
        "bp_stage": stage,
        "known_hypertension": h.hypertension,
        "target": targets,
        "summary": (
            f"Blood pressure stage: {stage.replace('_', ' ')}." if stage else "Known hypertension; no current reading."
        ),
        "recommendations": recs,
    }
