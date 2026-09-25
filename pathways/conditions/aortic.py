"""Aortic emergency pathway — dissection and symptomatic aneurysm."""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "aortic"
NAME = "Aortic emergency"
DESCRIPTION = (
    "Aortic dissection or rupturing aneurysm. Abrupt, severe tearing or ripping "
    "pain that radiates to the back, often with very high blood pressure."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    cp, s, h, v = f.chest_pain, f.symptoms, f.history, f.vitals
    b = ScoreBuilder()

    if not f.has_chest_pain:
        return ConditionAssessment(ID, NAME, 0, Likelihood.UNLIKELY, Disposition.ROUTINE, summary="No chest pain reported.")

    b.add(cp.quality == "tearing", 35, "Tearing / ripping quality")
    b.add(cp.sudden_onset, 15, "Abrupt onset (maximal within seconds)",
          "Did the pain reach its worst intensity within seconds?")
    b.add(cp.worst_ever or (cp.severity or 0) >= 9, 12, "Severe / worst-ever pain")
    b.add(f.radiates_to("back"), 18, "Radiates to the back",
          "Does the pain go through to your back or between your shoulder blades?" if not cp.radiation else None)
    b.add(s.syncope, 8, "Fainting")
    b.add(s.focal_neuro_deficit, 10, "Neurological deficit (branch-vessel involvement)")
    b.add(h.hypertension, 6, "Hypertension")
    b.add(h.connective_tissue_disorder, 15, "Connective tissue disorder (e.g. Marfan)")
    b.add(h.known_aortic_aneurysm, 15, "Known aortic aneurysm")
    b.add(h.stimulant_or_cocaine_use, 6, "Stimulant or cocaine use")
    if f.age is not None and f.age >= 60:
        b.add(True, 4, f"Age {f.age}")
    if v.systolic_bp is not None and v.systolic_bp >= 160:
        b.add(True, 6, f"Systolic BP {v.systolic_bp} mmHg")

    b.subtract(cp.exertional and cp.relieved_by_rest, 10, "Exertional, rest-relieved pain favours angina")
    b.subtract(cp.reproducible_on_palpation, 8, "Pain reproduced on palpation")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    # Aortic dissection is lethal enough that "possible" already means ED.
    if likelihood in (Likelihood.HIGH, Likelihood.LIKELY):
        disposition = Disposition.EMERGENCY
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.URGENT
    else:
        disposition = Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "Presentation is classic for aortic dissection — immediate emergency care.",
        Likelihood.LIKELY: "Aortic dissection must be excluded urgently (CT angiography).",
        Likelihood.POSSIBLE: "Some features raise concern for an aortic cause.",
        Likelihood.UNLIKELY: "Aortic emergency unlikely.",
    }[likelihood]

    return ConditionAssessment(ID, NAME, score, likelihood, disposition, b.supporting, b.against, b.missing[:2], summary)
