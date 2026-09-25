"""Pericarditis / myocarditis pathway."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "pericarditis"
NAME = "Pericarditis / myocarditis"
DESCRIPTION = (
    "Inflammation of the heart lining or muscle, often after a viral illness. "
    "Sharp pain that is worse lying flat and eased by sitting forward, with fever."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    cp, s, h = f.chest_pain, f.symptoms, f.history
    b = ScoreBuilder()

    if not f.has_chest_pain:
        return ConditionAssessment(ID, NAME, 0, Likelihood.UNLIKELY, Disposition.ROUTINE, summary="No chest pain reported.")

    b.add(cp.positional, 28, "Worse lying flat, better sitting forward",
          "Is the pain worse when you lie down and better when you sit up and lean forward?")
    b.add(cp.pleuritic, 16, "Sharp pain worse with breathing",
          "Is the pain worse when you breathe in deeply?" if cp.positional is not False else None)
    b.add(cp.quality == "sharp", 10, "Sharp quality")
    b.add(s.recent_viral_illness, 18, "Recent viral / flu-like illness",
          "Have you had a cold, flu or other viral illness in the last few weeks?")
    b.add(s.fever, 12, "Fever", "Do you have a fever?")
    b.add(s.fatigue, 4, "Fatigue")
    b.add(s.dyspnea, 6, "Shortness of breath (possible myocarditis / effusion)")
    b.add(s.palpitations, 5, "Palpitations")
    if f.age is not None and f.age < 45:
        b.add(True, 6, "Younger age")
    if cp.duration_minutes is not None and cp.duration_minutes >= 60 * 6:
        b.add(True, 6, "Persistent pain over many hours")

    b.subtract(cp.exertional and cp.relieved_by_rest, 12, "Exertional, rest-relieved pain favours angina")
    b.subtract(cp.quality == "pressure" and not cp.positional, 6, "Pressure quality without positional change")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    # Myocarditis can cause arrhythmia / HF; ECG and troponin are needed soon.
    if likelihood == Likelihood.HIGH and (s.dyspnea_at_rest or s.syncope or s.palpitations):
        disposition = Disposition.EMERGENCY
    elif likelihood in (Likelihood.HIGH, Likelihood.LIKELY):
        disposition = Disposition.URGENT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.PROMPT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "Strong features of pericarditis; myocarditis must be considered.",
        Likelihood.LIKELY: "Pericarditis is likely — needs ECG, troponin and inflammatory markers.",
        Likelihood.POSSIBLE: "Some inflammatory features present.",
        Likelihood.UNLIKELY: "Pericarditis / myocarditis unlikely.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:3], summary)
