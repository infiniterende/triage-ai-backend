"""
Acute coronary syndrome / myocardial ischemia pathway.

Heuristic weights are informed by the HEART score components (History, Age,
Risk factors) and the classic typical-angina triad. It is a triage aid, not a
diagnostic model — anything "likely" or above needs an ECG and troponin.
"""

from __future__ import annotations

from ..disposition import Disposition
from ..findings import ClinicalFindings
from .base import ConditionAssessment, Likelihood, ScoreBuilder, likelihood_from_score

ID = "acs"
NAME = "Acute coronary syndrome / ischemia"
DESCRIPTION = (
    "Heart attack and unstable angina. Pressure-type chest pain, exertional "
    "trigger, radiation to arm or jaw, sweating and cardiovascular risk factors."
)


def assess(f: ClinicalFindings) -> ConditionAssessment:
    cp, s, h = f.chest_pain, f.symptoms, f.history
    b = ScoreBuilder()

    if not f.has_chest_pain and not (s.dyspnea and (h.diabetes or (f.age or 0) >= 65)):
        # Anginal-equivalent presentations (diabetics, elderly) can be painless.
        return ConditionAssessment(
            pathway=ID,
            name=NAME,
            score=0,
            likelihood=Likelihood.UNLIKELY,
            disposition=Disposition.ROUTINE,
            summary="No chest pain or anginal equivalent reported.",
            missing=[] if f.chest_pain.present is False else ["Are you having any chest pain, pressure or tightness?"],
        )

    # --- Pain character (typical angina triad) --------------------------------
    b.add(cp.quality == "pressure", 22, "Pressure / squeezing / tightness quality",
          "How would you describe the pain — pressure, sharp, burning or tearing?" if cp.quality is None else None,
          negative_label="Non-pressure quality" if cp.quality in {"sharp", "pleuritic", "burning"} else None)
    b.add(cp.location == "substernal", 8, "Substernal location",
          "Where exactly is the pain — centre of the chest, left, right or upper stomach?" if cp.location is None else None)
    b.add(cp.exertional, 14, "Brought on by exertion or emotional stress",
          "Does the pain come on with physical activity or stress?")
    b.add(cp.relieved_by_rest, 8, "Relieved by rest",
          "Does it ease within a few minutes of resting?")
    b.add(cp.relieved_by_nitroglycerin, 6, "Relieved by nitroglycerin")
    b.add(f.radiates_to("arm", "left_arm", "both_arms"), 12, "Radiates to the arm(s)",
          "Does the pain spread to your arm, jaw, neck or back?" if not cp.radiation and cp.present else None)
    b.add(f.radiates_to("jaw", "neck"), 8, "Radiates to jaw or neck")

    # --- Associated symptoms ----------------------------------------------------
    b.add(s.diaphoresis, 10, "Sweating with the pain", "Are you sweating or clammy?")
    b.add(s.nausea or s.vomiting, 5, "Nausea or vomiting", "Any nausea?")
    b.add(s.dyspnea, 6, "Shortness of breath", "Are you short of breath?")

    # --- Duration / course --------------------------------------------------------
    if cp.duration_minutes is not None:
        if cp.duration_minutes >= 20:
            b.add(True, 8, "Pain lasting 20 minutes or more")
        elif cp.duration_minutes < 2:
            b.subtract(True, 8, "Very brief (seconds-long) pain is atypical for ischemia")
    else:
        b.missing.append("How long does each episode of pain last?")

    # --- Age, sex, risk factors, prior CAD -------------------------------------------
    if f.age is not None:
        if f.age >= 65:
            b.add(True, 10, f"Age {f.age}")
        elif f.age >= 45:
            b.add(True, 5, f"Age {f.age}")
        elif f.age < 30:
            b.subtract(True, 8, "Young age lowers pre-test probability")
    else:
        b.missing.append("How old are you?")
    if f.is_male:
        b.add(True, 4, "Male sex")
    rf = f.risk_factor_count()
    if rf >= 3:
        b.add(True, 10, f"{rf} cardiovascular risk factors")
    elif rf >= 1:
        b.add(True, 5, f"{rf} cardiovascular risk factor(s)")
    if h.hypertension is None and h.diabetes is None and h.smoking is None:
        b.missing.append("Do you have high blood pressure, diabetes, high cholesterol, or do you smoke?")
    b.add(f.known_cad(), 15, "Known coronary artery disease / prior heart attack or stent",
          "Have you ever been told you have heart disease, or had a heart attack, stent or bypass?"
          if h.coronary_artery_disease is None and h.prior_myocardial_infarction is None else None)
    b.add(h.stimulant_or_cocaine_use, 8, "Recent stimulant or cocaine use")

    # --- Features that argue against ischemia --------------------------------------
    b.subtract(cp.reproducible_on_palpation, 12, "Pain reproduced by pressing on the chest")
    b.subtract(cp.pleuritic, 8, "Sharp pain worse with breathing")
    b.subtract(cp.positional, 6, "Pain changes with position")

    score = b.clamp()
    likelihood = likelihood_from_score(score)

    ongoing = cp.ongoing is not False
    if likelihood == Likelihood.HIGH or (likelihood == Likelihood.LIKELY and ongoing):
        disposition = Disposition.EMERGENCY
    elif likelihood == Likelihood.LIKELY:
        disposition = Disposition.URGENT
    elif likelihood == Likelihood.POSSIBLE:
        disposition = Disposition.URGENT if ongoing and (cp.severity or 0) >= 6 else Disposition.PROMPT
    else:
        disposition = Disposition.PROMPT if f.has_chest_pain else Disposition.ROUTINE

    summary = {
        Likelihood.HIGH: "Presentation is highly suggestive of acute coronary syndrome.",
        Likelihood.LIKELY: "Several features of myocardial ischemia are present; ECG and troponin needed.",
        Likelihood.POSSIBLE: "Some ischemic features; cannot exclude ACS without testing.",
        Likelihood.UNLIKELY: "Few ischemic features; ACS is unlikely but chest pain still merits review.",
    }[likelihood]

    return ConditionAssessment(
        pathway=ID,
        name=NAME,
        score=score,
        likelihood=likelihood,
        disposition=disposition,
        supporting=b.supporting,
        against=b.against,
        missing=b.missing[:4],
        summary=summary,
    )
