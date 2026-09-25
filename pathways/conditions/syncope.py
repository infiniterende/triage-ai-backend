"""Syncope / structural heart disease pathway."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "syncope"
NAME = "Syncope / structural disease"
DESCRIPTION = (
    "Fainting or near-fainting that may be cardiac in origin, including valve "
    "disease and cardiomyopathy. Exertional syncope and a family history of "
    "sudden death are the most worrying features."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    s, h = f.symptoms, f.history
    b = ScoreBuilder()

    b.add(s.syncope, 30, "Fainting (syncope)", "Have you fainted or blacked out?")
    b.add(s.presyncope, 12, "Near-fainting", "Have you felt like you were about to faint?" if s.syncope is None else None)
    b.add(s.exertional_syncope, 25, "Fainting during exertion",
          "Did the fainting happen during exercise or exertion?" if s.syncope else None)
    b.add(s.syncope and f.has_chest_pain, 12, "Fainting with chest pain")
    b.add(s.syncope and s.palpitations, 12, "Fainting with palpitations")
    b.add(h.family_history_sudden_death, 18, "Family history of sudden cardiac death",
          "Has anyone in your family died suddenly at a young age?" if s.syncope else None)
    b.add(h.hypertrophic_cardiomyopathy, 18, "Known hypertrophic cardiomyopathy")
    b.add(h.valvular_disease_or_murmur, 12, "Known valve disease / heart murmur",
          "Have you been told you have a heart murmur or valve problem?" if s.syncope else None)
    b.add(h.heart_failure or f.known_cad(), 8, "Structural heart disease")
    if f.age is not None and f.age >= 65 and s.syncope:
        b.add(True, 8, f"Age {f.age}")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    if s.exertional_syncope or (s.syncope and (f.has_chest_pain or s.palpitations)):
        disposition = Disposition.EMERGENCY
    elif s.syncope and likelihood in (Likelihood.HIGH, Likelihood.LIKELY):
        disposition = Disposition.URGENT
    elif s.syncope:
        disposition = Disposition.PROMPT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.PROMPT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "High-risk syncope — cardiac cause must be excluded urgently.",
        Likelihood.LIKELY: "Syncope with features suggesting a cardiac cause; ECG and echo needed.",
        Likelihood.POSSIBLE: "Near-fainting or low-risk syncope — clinical review advised.",
        Likelihood.UNLIKELY: "No syncope reported.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:3], summary)
