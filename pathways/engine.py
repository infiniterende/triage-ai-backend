"""
Pipeline orchestration: findings → red flags → router → disposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .conditions import ConditionAssessment, Likelihood
from .conditions import cad_risk, diabetes, hypertensive
from .disposition import (
    Disposition,
    DispositionRecommendation,
    build_recommendation,
    max_disposition,
)
from .findings import ClinicalFindings
from .red_flags import RedFlag, detect_red_flags, screening_questions
from .router import route

ENGINE_VERSION = "1.0.0"


@dataclass
class PathwayResult:
    findings: ClinicalFindings
    red_flags: List[RedFlag]
    assessments: List[ConditionAssessment]  # ranked, primary first
    chronic: List[Dict[str, Any]]
    disposition: DispositionRecommendation
    next_questions: List[str]
    risk_percent: Optional[int]  # headline number for dashboards
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engine_version: str = ENGINE_VERSION

    @property
    def primary(self) -> Optional[ConditionAssessment]:
        return self.assessments[0] if self.assessments else None

    def to_dict(self) -> Dict[str, Any]:
        primary = self.primary
        return {
            "engine_version": self.engine_version,
            "evaluated_at": self.evaluated_at,
            "findings": self.findings.to_dict(),
            "red_flags": [r.to_dict() for r in self.red_flags],
            "primary_pathway": primary.to_dict() if primary else None,
            "pathways": [a.to_dict() for a in self.assessments],
            "chronic_pathways": self.chronic,
            "disposition": self.disposition.to_dict(),
            "next_questions": self.next_questions,
            "risk_percent": self.risk_percent,
        }


def _headline_risk(primary: Optional[ConditionAssessment], cad: Dict[str, Any], red_flags: List[RedFlag]) -> Optional[int]:
    """
    A single 0–100 number for dashboards. Uses the CAD model when it applies,
    otherwise the primary pathway's evidence score; red flags floor it at 85.
    """
    candidates = []
    if cad.get("probability") is not None:
        candidates.append(int(cad["probability"]))
    if primary and primary.likelihood != Likelihood.UNLIKELY:
        candidates.append(primary.score)
    if any(r.severity == "emergency" for r in red_flags):
        candidates.append(85)
    return max(candidates) if candidates else (primary.score if primary else None)


def _clinician_summary(f: ClinicalFindings, primary: Optional[ConditionAssessment], red_flags: List[RedFlag], level: Disposition, cad: Dict[str, Any]) -> str:
    parts = []
    demo = " ".join(x for x in (f"{f.age}y" if f.age else None, f.sex) if x)
    if demo:
        parts.append(demo.capitalize())
    if f.has_chest_pain:
        cp = f.chest_pain
        desc = ", ".join(
            x
            for x in (
                cp.quality,
                cp.location,
                "radiating to " + "/".join(cp.radiation) if cp.radiation else None,
                "exertional" if cp.exertional else None,
                f"{cp.duration_minutes} min" if cp.duration_minutes else None,
                "ongoing" if cp.ongoing else None,
            )
            if x
        )
        parts.append(f"chest pain ({desc})" if desc else "chest pain")
    sx = [
        label
        for cond, label in (
            (f.symptoms.dyspnea, "dyspnea"),
            (f.symptoms.diaphoresis, "diaphoresis"),
            (f.symptoms.nausea, "nausea"),
            (f.symptoms.palpitations, "palpitations"),
            (f.symptoms.syncope, "syncope"),
            (f.symptoms.orthopnea, "orthopnea"),
            (f.symptoms.leg_swelling, "leg swelling"),
        )
        if cond
    ]
    if sx:
        parts.append("with " + ", ".join(sx))
    rf = [
        label
        for cond, label in (
            (f.history.hypertension, "HTN"),
            (f.history.diabetes, "DM"),
            (f.history.hyperlipidemia, "HLD"),
            (f.history.smoking, "smoker"),
            (f.known_cad(), "known CAD"),
            (f.history.heart_failure, "HF"),
            (f.history.atrial_fibrillation, "AF"),
        )
        if cond
    ]
    if rf:
        parts.append("RF: " + ", ".join(rf))
    sentence = "; ".join(parts) + "." if parts else "Limited history available."
    if red_flags:
        sentence += " RED FLAGS: " + "; ".join(r.label for r in red_flags) + "."
    if primary and primary.likelihood != Likelihood.UNLIKELY:
        sentence += f" Leading pathway: {primary.name} ({primary.likelihood.value}, score {primary.score})."
    if cad.get("probability") is not None:
        sentence += f" CAD pre-test probability {cad['probability']}%."
    sentence += f" Disposition: {level.value.upper()}."
    return sentence


def run_pathway(findings: ClinicalFindings) -> PathwayResult:
    # 1. Safety / red-flag layer ------------------------------------------------
    red_flags = detect_red_flags(findings)

    # 2. Cardiovascular triage router → condition-specific assessments ----------
    assessments = route(findings)
    primary = assessments[0] if assessments else None

    # 3. Chronic pathways run alongside -------------------------------------------
    cad = cad_risk.assess_chronic(findings)
    chronic = [cad, hypertensive.assess_chronic(findings), diabetes.assess_chronic(findings)]

    # 4. Disposition -------------------------------------------------------------
    reasons: List[str] = []
    level = Disposition.ROUTINE
    for flag in red_flags:
        level = max_disposition(level, Disposition.EMERGENCY if flag.severity == "emergency" else Disposition.URGENT)
        reasons.append(f"Red flag: {flag.label}")
    for a in assessments:
        if a.likelihood == Likelihood.UNLIKELY:
            continue
        if a.disposition.rank > Disposition.ROUTINE.rank:
            reasons.append(f"{a.name}: {a.likelihood.value} ({a.disposition.value})")
        level = max_disposition(level, a.disposition)
    # Chest pain of any kind should not simply be dismissed.
    if findings.has_chest_pain and level == Disposition.ROUTINE:
        level = Disposition.PROMPT
        reasons.append("Chest pain warrants clinician review even when low risk")
    if cad.get("band") in ("high", "very_high") and level.rank < Disposition.PROMPT.rank:
        level = Disposition.PROMPT
        reasons.append(f"High CAD pre-test probability ({cad['probability']}%)")

    summary = _clinician_summary(findings, primary, red_flags, level, cad)
    recommendation = build_recommendation(level, reasons, summary)

    # 5. Questions the agent should ask next (red-flag screening first) ----------
    next_questions: List[str] = []
    for q in screening_questions(findings):
        if q not in next_questions:
            next_questions.append(q)
    for a in assessments[:2]:
        if a.likelihood == Likelihood.UNLIKELY and a is not primary:
            continue
        for q in a.missing:
            if q not in next_questions:
                next_questions.append(q)
    if level == Disposition.EMERGENCY:
        next_questions = []  # stop interviewing; tell the patient to call 911

    return PathwayResult(
        findings=findings,
        red_flags=red_flags,
        assessments=assessments,
        chronic=chronic,
        disposition=recommendation,
        next_questions=next_questions[:5],
        risk_percent=_headline_risk(primary, cad, red_flags),
    )


def evaluate(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper: JSON in, JSON out."""
    return run_pathway(ClinicalFindings.from_dict(data)).to_dict()
