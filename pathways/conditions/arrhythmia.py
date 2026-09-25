"""Arrhythmia pathway — palpitations, irregular rhythm, rate extremes, AF."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "arrhythmia"
NAME = "Arrhythmia"
DESCRIPTION = (
    "Abnormal heart rhythms including atrial fibrillation, SVT and "
    "bradyarrhythmias. Palpitations, irregular heartbeat, dizziness or fainting."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    s, h, v = f.symptoms, f.history, f.vitals
    b = ScoreBuilder()

    b.add(s.palpitations, 25, "Palpitations", "Do you feel your heart racing, pounding or skipping?")
    b.add(s.irregular_heartbeat, 20, "Irregular heartbeat",
          "Does the heartbeat feel irregular, like it is skipping or fluttering?" if s.palpitations else None)
    b.add(s.presyncope or s.dizziness, 12, "Light-headedness or dizziness",
          "Do you feel light-headed or dizzy with it?")
    b.add(s.syncope, 20, "Fainting", "Have you fainted?")
    b.add(h.atrial_fibrillation or h.arrhythmia, 15, "Known arrhythmia / atrial fibrillation",
          "Have you ever been told you have an irregular heart rhythm?")
    b.add(h.pacemaker_or_icd, 6, "Pacemaker or ICD in place")
    b.add(h.stimulant_or_cocaine_use, 8, "Stimulant or cocaine use")
    b.add(h.heart_failure or f.known_cad(), 6, "Structural heart disease increases arrhythmia risk")

    unstable = False
    if v.heart_rate is not None:
        if v.heart_rate > 150 or v.heart_rate < 40:
            b.add(True, 25, f"Heart rate {v.heart_rate} bpm")
            unstable = True
        elif v.heart_rate > 120 or v.heart_rate < 50:
            b.add(True, 12, f"Heart rate {v.heart_rate} bpm")
    elif s.palpitations:
        b.missing.append("If you can, count your pulse for 30 seconds — how fast is it?")
    if v.systolic_bp is not None and v.systolic_bp < 90:
        unstable = True

    if s.palpitations and f.has_chest_pain:
        b.add(True, 8, "Palpitations with chest pain")
    if s.palpitations and s.dyspnea:
        b.add(True, 6, "Palpitations with breathlessness")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    if unstable or (s.syncope and s.palpitations):
        disposition = Disposition.EMERGENCY
    elif likelihood in (Likelihood.HIGH, Likelihood.LIKELY) and (f.has_chest_pain or s.dyspnea or s.presyncope):
        disposition = Disposition.URGENT
    elif likelihood in (Likelihood.HIGH, Likelihood.LIKELY):
        disposition = Disposition.PROMPT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.PROMPT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "Symptoms strongly suggest a clinically significant arrhythmia.",
        Likelihood.LIKELY: "Palpitations with associated symptoms — needs an ECG / rhythm monitoring.",
        Likelihood.POSSIBLE: "Some rhythm-related symptoms; ECG advisable.",
        Likelihood.UNLIKELY: "No significant rhythm symptoms reported.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:3], summary)
