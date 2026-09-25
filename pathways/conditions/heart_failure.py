"""Heart failure pathway — new or decompensated congestive heart failure."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "heart_failure"
NAME = "Heart failure"
DESCRIPTION = (
    "New or worsening heart failure. Breathlessness on exertion or lying flat, "
    "waking at night gasping, leg swelling, rapid weight gain and fatigue."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    s, h = f.symptoms, f.history
    b = ScoreBuilder()

    b.add(s.orthopnea, 22, "Breathless lying flat (orthopnea)",
          "Do you need to prop yourself up on pillows to breathe at night?" if s.dyspnea else None)
    b.add(s.paroxysmal_nocturnal_dyspnea, 18, "Waking at night gasping for breath",
          "Do you wake up at night suddenly short of breath?" if s.dyspnea else None)
    b.add(s.leg_swelling and not s.unilateral_leg_swelling, 16, "Swelling of both legs / ankles",
          "Are your ankles or legs swollen?" if s.dyspnea else None)
    b.add(s.rapid_weight_gain, 10, "Rapid weight gain (fluid retention)")
    b.add(s.dyspnea, 12, "Shortness of breath", "Are you short of breath?")
    b.add(s.dyspnea_at_rest, 12, "Breathless at rest")
    b.add(s.fatigue, 5, "Fatigue / reduced exercise tolerance")
    b.add(s.cough, 3, "Cough (may be fluid-related)")
    b.add(h.heart_failure, 22, "Known heart failure",
          "Have you ever been told you have heart failure or a weak heart?" if s.dyspnea else None)
    b.add(f.known_cad(), 8, "Prior coronary disease / heart attack")
    b.add(h.hypertension, 4, "Hypertension")
    b.add(h.diabetes, 3, "Diabetes")
    b.add(h.valvular_disease_or_murmur, 6, "Valve disease")
    b.add(h.atrial_fibrillation, 5, "Atrial fibrillation")
    if f.vitals.spo2 is not None and f.vitals.spo2 < 94:
        b.add(True, 10, f"Oxygen saturation {f.vitals.spo2}%")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    if s.dyspnea_at_rest and likelihood in (Likelihood.HIGH, Likelihood.LIKELY):
        disposition = Disposition.EMERGENCY
    elif likelihood == Likelihood.HIGH:
        disposition = Disposition.URGENT
    elif likelihood == Likelihood.LIKELY:
        disposition = Disposition.URGENT if s.orthopnea else Disposition.PROMPT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.PROMPT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "Strong picture of decompensated heart failure.",
        Likelihood.LIKELY: "Congestive symptoms present; needs examination, BNP and echo.",
        Likelihood.POSSIBLE: "Some congestive features — worth a clinical review.",
        Likelihood.UNLIKELY: "No significant heart-failure symptoms reported.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:3], summary)
