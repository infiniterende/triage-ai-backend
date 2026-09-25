"""
Diabetes cardiometabolic pathway (chronic).

Diabetes doubles cardiovascular risk and blunts typical chest-pain symptoms, so
the acute pathways already up-weight it. This chronic module surfaces the
long-term management context for the patient and clinician dashboards.
"""

from __future__ import annotations

from typing import Any, Dict

from ..findings import ClinicalFindings

ID = "diabetes_cardiometabolic"
NAME = "Diabetes & cardiometabolic risk"
DESCRIPTION = (
    "Diabetes-driven cardiovascular risk: silent ischemia, atypical presentations "
    "and the need for tight blood-pressure, lipid and glucose control."
)


def assess_chronic(f: ClinicalFindings) -> Dict[str, Any]:
    h, s = f.history, f.symptoms
    if not h.diabetes:
        return {
            "pathway": ID,
            "name": NAME,
            "status": "not_applicable" if h.diabetes is False else "unknown",
            "summary": "No diabetes reported." if h.diabetes is False else "Diabetes status not recorded.",
            "recommendations": [],
        }

    notes = []
    if s.dyspnea and not f.has_chest_pain:
        notes.append("Breathlessness without chest pain can be an anginal equivalent in diabetes — treated as a possible ischemic presentation.")
    if f.known_cad():
        notes.append("Established coronary disease with diabetes: very high cardiovascular risk category.")
    comorbid = [
        label
        for cond, label in (
            (h.hypertension, "hypertension"),
            (h.hyperlipidemia, "high cholesterol"),
            (h.smoking, "smoking"),
            (h.chronic_kidney_disease, "chronic kidney disease"),
        )
        if cond
    ]
    recs = [
        "Blood pressure target below 130/80 mmHg",
        "Statin therapy is recommended for most adults with diabetes over 40",
        "HbA1c review at least every 6 months; individualised glucose target",
        "Annual kidney function, urine albumin, eye and foot checks",
        "Consider SGLT2 inhibitor or GLP-1 RA if cardiovascular disease is present",
    ]
    if h.smoking:
        recs.insert(0, "Smoking cessation is the single most effective risk reduction")
    return {
        "pathway": ID,
        "name": NAME,
        "status": "active",
        "comorbidities": comorbid,
        "summary": "Diabetes present" + (f" with {', '.join(comorbid)}" if comorbid else "") + " — elevated long-term cardiovascular risk.",
        "notes": notes,
        "recommendations": recs,
    }
