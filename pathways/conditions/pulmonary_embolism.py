"""Pulmonary embolism pathway — Wells-style clinical probability."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "pulmonary_embolism"
NAME = "Pulmonary embolism"
DESCRIPTION = (
    "Blood clot in the lung. Sudden breathlessness, sharp pain worse on "
    "breathing, fast heart rate, coughing blood, a swollen calf, or recent "
    "surgery, immobility or clotting history."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    cp, s, h, v = f.chest_pain, f.symptoms, f.history, f.vitals
    b = ScoreBuilder()

    b.add(cp.pleuritic, 18, "Pleuritic chest pain (worse on breathing)",
          "Is the pain worse when you take a deep breath?" if f.has_chest_pain else None)
    b.add(cp.sudden_onset and s.dyspnea, 14, "Sudden-onset breathlessness")
    b.add(s.dyspnea, 10, "Shortness of breath", "Are you short of breath?")
    b.add(s.dyspnea_at_rest, 6, "Breathless at rest")
    b.add(s.hemoptysis, 15, "Coughing up blood", "Have you coughed up any blood?" if s.dyspnea else None)
    b.add(s.unilateral_leg_swelling, 18, "One swollen, painful calf (possible DVT)",
          "Is one of your calves swollen or painful?" if s.dyspnea or cp.pleuritic else None)
    b.add(h.prior_pe_or_dvt, 15, "Previous clot (DVT / PE)",
          "Have you ever had a blood clot in your leg or lung?" if s.dyspnea else None)
    b.add(h.recent_surgery_or_immobilization, 14, "Recent surgery, injury or immobility",
          "Any surgery, long travel or bed rest in the last month?" if s.dyspnea else None)
    b.add(h.active_cancer, 10, "Active cancer")
    b.add(h.pregnancy_or_estrogen_use, 8, "Pregnancy or estrogen-containing medication")
    b.add(s.syncope, 8, "Fainting")
    if v.heart_rate is not None and v.heart_rate > 100:
        b.add(True, 12, f"Heart rate {v.heart_rate} bpm (tachycardia)")
    if v.spo2 is not None and v.spo2 < 94:
        b.add(True, 10, f"Oxygen saturation {v.spo2}%")

    b.subtract(cp.quality == "pressure" and cp.exertional, 8, "Exertional pressure-type pain favours ischemia")
    b.subtract(cp.reproducible_on_palpation, 8, "Pain reproduced on palpation")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    if likelihood == Likelihood.HIGH or (likelihood == Likelihood.LIKELY and (s.dyspnea_at_rest or s.syncope or (v.spo2 or 100) < 92)):
        disposition = Disposition.EMERGENCY
    elif likelihood == Likelihood.LIKELY:
        disposition = Disposition.URGENT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.URGENT if s.dyspnea else Disposition.PROMPT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "High clinical probability of pulmonary embolism.",
        Likelihood.LIKELY: "Pulmonary embolism is a real possibility; needs D-dimer / CT in an emergency setting.",
        Likelihood.POSSIBLE: "Some features of PE; low-to-intermediate probability.",
        Likelihood.UNLIKELY: "Few features of pulmonary embolism.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:3], summary)
