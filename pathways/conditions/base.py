from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..disposition import Disposition


class Likelihood(str, Enum):
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"unlikely": 0, "possible": 1, "likely": 2, "high": 3}[self.value]


def likelihood_from_score(score: float) -> Likelihood:
    if score >= 70:
        return Likelihood.HIGH
    if score >= 45:
        return Likelihood.LIKELY
    if score >= 20:
        return Likelihood.POSSIBLE
    return Likelihood.UNLIKELY


@dataclass
class ConditionAssessment:
    pathway: str  # machine id, e.g. "acs"
    name: str  # human label
    score: int  # 0–100 relative weight of evidence
    likelihood: Likelihood
    disposition: Disposition  # what this pathway alone would recommend
    supporting: List[str] = field(default_factory=list)  # evidence for
    against: List[str] = field(default_factory=list)  # evidence against
    missing: List[str] = field(default_factory=list)  # questions to ask next
    summary: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)  # e.g. numeric scores

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["likelihood"] = self.likelihood.value
        d["disposition"] = self.disposition.value
        return d


class ScoreBuilder:
    """Small helper so each pathway reads like a checklist."""

    def __init__(self) -> None:
        self.score = 0.0
        self.supporting: List[str] = []
        self.against: List[str] = []
        self.missing: List[str] = []

    def add(self, condition: Optional[bool], points: float, label: str, question: Optional[str] = None, negative_label: Optional[str] = None) -> "ScoreBuilder":
        """
        ``condition`` True → add points and record supporting evidence.
        False → optionally record evidence against.
        None  → record the question the agent should ask.
        """
        if condition:
            self.score += points
            self.supporting.append(label)
        elif condition is False and negative_label:
            self.against.append(negative_label)
        elif condition is None and question:
            self.missing.append(question)
        return self

    def subtract(self, condition: Optional[bool], points: float, label: str) -> "ScoreBuilder":
        if condition:
            self.score -= points
            self.against.append(label)
        return self

    def clamp(self) -> int:
        return int(max(0, min(100, round(self.score))))
