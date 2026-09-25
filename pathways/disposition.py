"""
Disposition levels shared by the red-flag layer, the condition modules and the
final recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List


class Disposition(str, Enum):
    EMERGENCY = "emergency"  # call 911 now
    URGENT = "urgent"  # emergency department / urgent care today
    PROMPT = "prompt"  # clinician within 24–72 hours
    ROUTINE = "routine"  # routine appointment within 1–2 weeks

    @property
    def rank(self) -> int:
        return {"emergency": 3, "urgent": 2, "prompt": 1, "routine": 0}[self.value]


def max_disposition(*levels: Disposition) -> Disposition:
    return max(levels, key=lambda d: d.rank, default=Disposition.ROUTINE)


DISPOSITION_META: Dict[Disposition, Dict[str, Any]] = {
    Disposition.EMERGENCY: {
        "headline": "Call 911 now",
        "timeframe": "Immediately",
        "patient_message": (
            "Your symptoms could be a medical emergency. Please call 911 (or your "
            "local emergency number) right now. Do not drive yourself. If you "
            "have aspirin and are not allergic, emergency dispatch may advise you "
            "to chew one while you wait."
        ),
        "actions": [
            "Call 911 or have someone call for you",
            "Do not drive yourself to hospital",
            "Unlock the door and sit or lie down while you wait",
            "Tell the dispatcher about your chest symptoms and medical history",
        ],
        "scheduling": None,
    },
    Disposition.URGENT: {
        "headline": "Go to an emergency department today",
        "timeframe": "Within the next few hours",
        "patient_message": (
            "Your symptoms need to be checked in person today, ideally in an "
            "emergency department where an ECG and blood tests can be done. If "
            "they get worse at any point, call 911."
        ),
        "actions": [
            "Go to the nearest emergency department or urgent care today",
            "Have someone drive you — do not drive yourself",
            "Bring a list of your medications",
            "Call 911 if symptoms worsen on the way",
        ],
        "scheduling": "same_day",
    },
    Disposition.PROMPT: {
        "headline": "See a clinician within 1–3 days",
        "timeframe": "Within 24–72 hours",
        "patient_message": (
            "Your symptoms should be evaluated by a clinician soon — within the "
            "next one to three days. We can help you book an appointment. Seek "
            "emergency care right away if the symptoms return more severely, "
            "last longer, or come with sweating, fainting or breathlessness."
        ),
        "actions": [
            "Book an appointment with your doctor or a cardiologist within 72 hours",
            "Keep a note of when symptoms happen and what triggers them",
            "Avoid strenuous activity until you have been seen",
            "Call 911 if symptoms become severe or persistent",
        ],
        "scheduling": "within_72_hours",
    },
    Disposition.ROUTINE: {
        "headline": "Schedule a routine appointment",
        "timeframe": "Within 1–2 weeks",
        "patient_message": (
            "Nothing you've described points to an emergency, but symptoms like "
            "these are worth discussing with your doctor. We can help you "
            "schedule a routine visit. If anything changes or worsens, come back "
            "and reassess."
        ),
        "actions": [
            "Schedule a routine appointment within the next two weeks",
            "Track your symptoms, blood pressure and activity in the meantime",
            "Reassess with Agilance if symptoms change",
        ],
        "scheduling": "routine",
    },
}


@dataclass
class DispositionRecommendation:
    level: Disposition
    headline: str
    timeframe: str
    patient_message: str
    actions: List[str]
    scheduling: Any
    reasons: List[str] = field(default_factory=list)
    clinician_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


def build_recommendation(level: Disposition, reasons: List[str], clinician_summary: str) -> DispositionRecommendation:
    meta = DISPOSITION_META[level]
    return DispositionRecommendation(
        level=level,
        headline=meta["headline"],
        timeframe=meta["timeframe"],
        patient_message=meta["patient_message"],
        actions=list(meta["actions"]),
        scheduling=meta["scheduling"],
        reasons=reasons,
        clinician_summary=clinician_summary,
    )
