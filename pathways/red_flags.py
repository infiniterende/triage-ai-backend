"""
Safety / red-flag layer.

Runs *before* any pathway routing. Any "emergency" red flag short-circuits the
disposition to EMERGENCY regardless of what the condition-specific assessment
concludes. "Urgent" red flags raise the floor to URGENT.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .findings import ClinicalFindings


@dataclass(frozen=True)
class RedFlag:
    code: str
    label: str
    rationale: str
    severity: str  # "emergency" | "urgent"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_red_flags(f: ClinicalFindings) -> List[RedFlag]:
    flags: List[RedFlag] = []
    cp, s, h, v = f.chest_pain, f.symptoms, f.history, f.vitals

    def add(code: str, label: str, rationale: str, severity: str = "emergency") -> None:
        flags.append(RedFlag(code, label, rationale, severity))

    # --- Ischemic-type pain that is ongoing --------------------------------
    ischemic_quality = cp.quality == "pressure"
    ischemic_features = sum(
        1
        for x in (
            ischemic_quality,
            f.radiates_to("arm", "left_arm", "both_arms", "jaw", "neck"),
            s.diaphoresis,
            s.nausea,
            s.dyspnea,
        )
        if x
    )
    if f.has_chest_pain and cp.ongoing and ischemic_features >= 2:
        add(
            "ongoing_ischemic_pain",
            "Ongoing chest pain with ischemic features",
            "Pressure-type or radiating chest pain that is still present, with "
            "sweating, nausea or breathlessness, may represent an acute coronary "
            "syndrome.",
        )
    if f.has_chest_pain and cp.duration_minutes and cp.duration_minutes >= 20 and (
        ischemic_quality or f.known_cad()
    ) and cp.ongoing is not False:
        add(
            "prolonged_ischemic_pain",
            "Ischemic-type chest pain lasting 20 minutes or more",
            "Prolonged pressure-type pain, especially with known coronary disease, "
            "needs immediate evaluation for myocardial infarction.",
        )

    # --- Aortic ------------------------------------------------------------
    if f.has_chest_pain and (
        cp.quality == "tearing"
        or (cp.sudden_onset and cp.worst_ever and f.radiates_to("back"))
    ):
        add(
            "tearing_pain",
            "Sudden, severe tearing pain radiating to the back",
            "Classic presentation of aortic dissection — a time-critical emergency.",
        )

    # --- Breathing / oxygenation --------------------------------------------
    if s.dyspnea_at_rest and (f.has_chest_pain or s.syncope or s.hemoptysis):
        add(
            "dyspnea_at_rest",
            "Shortness of breath at rest with chest pain, fainting or coughing blood",
            "Suggests pulmonary embolism, heart failure or acute cardiac ischemia.",
        )
    if v.spo2 is not None and v.spo2 < 90:
        add(
            "hypoxia",
            f"Low oxygen saturation ({v.spo2}%)",
            "SpO₂ below 90% indicates respiratory or circulatory compromise.",
        )
    if s.hemoptysis and (s.dyspnea or f.has_chest_pain):
        add(
            "hemoptysis",
            "Coughing up blood with chest pain or breathlessness",
            "May indicate pulmonary embolism.",
            "urgent",
        )

    # --- Syncope -------------------------------------------------------------
    if s.syncope and (f.has_chest_pain or s.palpitations or s.exertional_syncope):
        add(
            "cardiac_syncope",
            "Fainting with chest pain, palpitations or during exertion",
            "Syncope with cardiac symptoms suggests a dangerous arrhythmia or "
            "structural heart disease.",
        )

    # --- Hemodynamics --------------------------------------------------------
    if v.heart_rate is not None and (v.heart_rate > 150 or v.heart_rate < 40):
        add(
            "extreme_heart_rate",
            f"Heart rate {v.heart_rate} bpm",
            "Very fast or very slow heart rate with symptoms can cause collapse.",
        )
    if v.systolic_bp is not None and v.systolic_bp < 90:
        add(
            "hypotension",
            f"Low blood pressure ({v.systolic_bp} mmHg systolic)",
            "Hypotension with cardiac symptoms suggests shock.",
        )
    end_organ = any(
        x
        for x in (
            s.severe_headache,
            s.vision_changes,
            s.confusion,
            s.focal_neuro_deficit,
            f.has_chest_pain,
            s.dyspnea_at_rest,
        )
    )
    if (
        v.systolic_bp is not None and v.systolic_bp >= 180
        or v.diastolic_bp is not None and v.diastolic_bp >= 120
    ) and end_organ:
        add(
            "hypertensive_emergency",
            "Severely elevated blood pressure with symptoms",
            "BP ≥ 180/120 with headache, vision change, confusion, chest pain or "
            "breathlessness is a hypertensive emergency.",
        )

    # --- Neurological --------------------------------------------------------
    if s.focal_neuro_deficit or s.confusion:
        add(
            "neuro_deficit",
            "New confusion, weakness, facial droop or speech difficulty",
            "Possible stroke or hypertensive encephalopathy — call 911.",
        )

    # --- Patient-reported severity ------------------------------------------
    if cp.worst_ever and f.has_chest_pain and cp.ongoing is not False:
        add(
            "worst_pain_ever",
            "Described as the worst pain ever experienced",
            "Extreme pain intensity warrants immediate emergency evaluation.",
            "urgent" if cp.quality in {"sharp", "pleuritic"} else "emergency",
        )

    # De-duplicate while keeping order.
    seen = set()
    unique: List[RedFlag] = []
    for flag in flags:
        if flag.code not in seen:
            seen.add(flag.code)
            unique.append(flag)
    return unique


def screening_questions(f: ClinicalFindings) -> List[str]:
    """Red-flag questions still unanswered — asked first by the agent."""
    qs: List[str] = []
    cp, s = f.chest_pain, f.symptoms
    if f.has_chest_pain:
        if cp.ongoing is None:
            qs.append("Is the chest pain happening right now?")
        if cp.worst_ever is None and (cp.severity is None or cp.severity >= 7):
            qs.append("Is this the worst pain you have ever felt?")
        if cp.sudden_onset is None:
            qs.append("Did the pain start suddenly, within seconds, or build up gradually?")
    if s.dyspnea_at_rest is None and (s.dyspnea or f.has_chest_pain):
        qs.append("Are you short of breath even while sitting still?")
    if s.syncope is None:
        qs.append("Have you fainted or nearly fainted?")
    if s.focal_neuro_deficit is None and (f.history.hypertension or f.age and f.age > 60):
        qs.append("Any new weakness, numbness, facial droop or trouble speaking?")
    return qs
