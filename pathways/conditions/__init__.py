"""
Condition-specific assessment modules.

Each acute pathway exposes ``assess(findings) -> ConditionAssessment``. The
router runs every acute pathway and ranks them; the chronic pathways run
alongside and never compete for "primary" — they enrich the result with
long-term risk context (CAD probability, hypertension stage, diabetes).
"""

from .base import ConditionAssessment, Likelihood, likelihood_from_score
from . import (
    acs,
    arrhythmia,
    heart_failure,
    hypertensive,
    pulmonary_embolism,
    aortic,
    pericarditis,
    syncope,
    cad_risk,
    diabetes,
)

ACUTE_PATHWAYS = [
    acs,
    arrhythmia,
    heart_failure,
    hypertensive,
    pulmonary_embolism,
    aortic,
    pericarditis,
    syncope,
]

CHRONIC_PATHWAYS = [cad_risk, hypertensive, diabetes]

__all__ = [
    "ConditionAssessment",
    "Likelihood",
    "likelihood_from_score",
    "ACUTE_PATHWAYS",
    "CHRONIC_PATHWAYS",
]
