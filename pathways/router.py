"""
Cardiovascular triage router.

Runs every acute condition pathway against the findings and ranks them. The
primary pathway is the one with the strongest evidence; ties are broken by the
more severe disposition so a dangerous diagnosis is never buried beneath a
benign one with the same score.
"""

from __future__ import annotations

from typing import Dict, List

from .conditions import ACUTE_PATHWAYS, ConditionAssessment
from .findings import ClinicalFindings


def route(f: ClinicalFindings) -> List[ConditionAssessment]:
    assessments = [module.assess(f) for module in ACUTE_PATHWAYS]
    assessments.sort(key=lambda a: (a.score, a.disposition.rank), reverse=True)
    return assessments


def catalog() -> List[Dict[str, str]]:
    """Static description of each acute pathway for documentation / UI."""
    return [
        {"id": m.ID, "name": m.NAME, "description": m.DESCRIPTION} for m in ACUTE_PATHWAYS
    ]
