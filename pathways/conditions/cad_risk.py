"""
Chronic coronary artery disease risk pathway.

Wraps the existing CAD Consortium clinical model (``calculate_cad_score``,
BMJ 2012) so the long-term pre-test probability of obstructive CAD is reported
alongside the acute triage. Missing inputs degrade gracefully to "insufficient
data" rather than a bogus number.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional

from ..findings import ClinicalFindings

# calculate_cad_score.py lives in the backend root; make it importable whether
# the engine is run from the package or as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:  # pragma: no cover - import shim
    from calculate_cad_score import cadc_clinical_risk as _cadc_clinical_risk  # type: ignore
except Exception:  # fallback keeps the engine usable standalone
    def _cadc_clinical_risk(age, male, chest_pain_type, diabetes=False, hypertension=False, dyslipidaemia=False, smoking=False):
        b0, b_age, b_male, b_atyp, b_typ = -7.539, 0.062, 1.332, 0.633, 1.998
        b_diab, b_htn, b_dlp, b_smk, b_interact = 0.828, 0.338, 0.422, 0.461, -0.402
        male = 1 if male else 0
        cp_atyp = 1 if chest_pain_type == "atypical" else 0
        cp_typ = 1 if chest_pain_type == "typical" else 0
        dm, htn, dlp, smk = (1 if diabetes else 0), (1 if hypertension else 0), (1 if dyslipidaemia else 0), (1 if smoking else 0)
        logit = (b0 + b_age * age + b_male * male + b_atyp * cp_atyp + b_typ * cp_typ
                 + b_diab * dm + b_htn * htn + b_dlp * dlp + b_smk * smk + b_interact * (dm * cp_typ))
        return 1 / (1 + math.exp(-logit))

ID = "cad_risk"
NAME = "Coronary artery disease risk"
DESCRIPTION = (
    "Pre-test probability of obstructive coronary artery disease from age, sex, "
    "chest-pain typicality and risk factors (CAD Consortium clinical model)."
)


def classify_chest_pain(f: ClinicalFindings) -> Optional[str]:
    """Typical / atypical / non-anginal using the Diamond–Forrester triad."""
    cp = f.chest_pain
    if not f.has_chest_pain:
        return None
    substernal = cp.location == "substernal" or cp.quality == "pressure"
    exertional = cp.exertional
    relieved = cp.relieved_by_rest or cp.relieved_by_nitroglycerin
    known = [x for x in (substernal, exertional, relieved) if x is not None]
    if len(known) < 2:
        return None
    count = sum(1 for x in (substernal, exertional, relieved) if x)
    if count == 3:
        return "typical"
    if count == 2:
        return "atypical"
    return "non-anginal"


def assess_chronic(f: ClinicalFindings) -> Dict[str, Any]:
    h = f.history
    missing = []
    if f.age is None:
        missing.append("age")
    if f.sex is None:
        missing.append("sex")
    pain_type = classify_chest_pain(f)
    if f.has_chest_pain and pain_type is None:
        missing.append("chest pain characteristics")

    result: Dict[str, Any] = {
        "pathway": ID,
        "name": NAME,
        "chest_pain_type": pain_type,
        "risk_factors": {
            "diabetes": h.diabetes,
            "hypertension": h.hypertension,
            "hyperlipidemia": h.hyperlipidemia,
            "smoking": h.smoking,
            "family_history": h.family_history_premature_cad,
        },
    }

    if missing or not f.has_chest_pain:
        result.update(
            status="insufficient_data" if missing else "not_applicable",
            probability=None,
            band=None,
            summary=(
                "Not enough information to estimate CAD probability (missing: " + ", ".join(missing) + ")."
                if missing
                else "CAD probability model applies to patients with chest pain."
            ),
            missing=missing,
        )
        return result

    p = _cadc_clinical_risk(
        age=f.age,
        male=bool(f.is_male),
        chest_pain_type=pain_type if pain_type != "non-anginal" else "non-specific",
        diabetes=bool(h.diabetes),
        hypertension=bool(h.hypertension),
        dyslipidaemia=bool(h.hyperlipidemia),
        smoking=bool(h.smoking),
    )
    pct = round(p * 100)
    if pct >= 85:
        band = "very_high"
    elif pct >= 50:
        band = "high"
    elif pct >= 15:
        band = "intermediate"
    else:
        band = "low"
    guidance = {
        "very_high": "Very high pre-test probability — treat as established CAD; invasive evaluation is often appropriate.",
        "high": "High pre-test probability — non-invasive ischemia testing or CT coronary angiography is recommended.",
        "intermediate": "Intermediate probability — outpatient non-invasive testing (CTCA or stress testing) is reasonable.",
        "low": "Low probability of obstructive CAD — routine risk-factor management; testing rarely changes care.",
    }[band]
    result.update(
        status="ok",
        probability=pct,
        band=band,
        summary=f"Estimated {pct}% pre-test probability of obstructive CAD ({band.replace('_', ' ')}).",
        guidance=guidance,
        missing=[],
    )
    return result
