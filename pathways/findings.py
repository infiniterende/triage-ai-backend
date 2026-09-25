"""
Structured clinical findings extracted from a patient conversation.

Every boolean is tri-state: ``True`` (reported present), ``False`` (reported
absent) or ``None`` (not yet asked / unknown). The distinction matters — the
router uses ``None`` fields to decide which condition-specific questions the
agent should ask next.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional

PAIN_QUALITIES = {"pressure", "sharp", "burning", "tearing", "dull", "pleuritic"}
PAIN_LOCATIONS = {"substernal", "left", "right", "epigastric", "diffuse"}
RADIATION_SITES = {"arm", "left_arm", "both_arms", "jaw", "neck", "back", "shoulder"}


def _to_bool(value: Any) -> Optional[bool]:
    """Coerce loosely-typed LLM output ('yes', 'No', 1, None) to a tri-state bool."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "present", "1", "positive"}:
        return True
    if text in {"false", "no", "n", "absent", "0", "negative", "denies"}:
        return False
    if text in {"unknown", "unsure", "none", "n/a", "na", "not asked"}:
        return None
    return None


def _to_int(value: Any) -> Optional[int]:
    """Parse '58', '58 years', '8/10', 8.0 → int; anything else → None."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return int(float(match.group())) if match else None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().split()[0])
    except (ValueError, IndexError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip().lower().replace(" ", "_") for p in value.split(",")]
        return [p for p in parts if p and p not in {"none", "no", "n/a"}]
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            s = _to_str(v)
            if s and s not in {"none", "no", "n/a"}:
                out.append(s.replace(" ", "_"))
        return out
    return []


@dataclass
class Vitals:
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    spo2: Optional[int] = None
    respiratory_rate: Optional[int] = None
    temperature_c: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Vitals":
        data = data or {}
        return cls(
            heart_rate=_to_int(data.get("heart_rate") or data.get("hr")),
            systolic_bp=_to_int(data.get("systolic_bp") or data.get("sbp")),
            diastolic_bp=_to_int(data.get("diastolic_bp") or data.get("dbp")),
            spo2=_to_int(data.get("spo2") or data.get("oxygen_saturation")),
            respiratory_rate=_to_int(data.get("respiratory_rate") or data.get("rr")),
            temperature_c=_to_float(data.get("temperature_c") or data.get("temp")),
        )


@dataclass
class ChestPain:
    present: Optional[bool] = None
    quality: Optional[str] = None  # one of PAIN_QUALITIES
    location: Optional[str] = None  # one of PAIN_LOCATIONS
    radiation: List[str] = field(default_factory=list)  # RADIATION_SITES
    ongoing: Optional[bool] = None
    duration_minutes: Optional[int] = None
    sudden_onset: Optional[bool] = None
    exertional: Optional[bool] = None
    relieved_by_rest: Optional[bool] = None
    relieved_by_nitroglycerin: Optional[bool] = None
    positional: Optional[bool] = None  # worse lying flat / better leaning forward
    pleuritic: Optional[bool] = None  # worse with deep breath
    reproducible_on_palpation: Optional[bool] = None
    severity: Optional[int] = None  # 0–10
    worst_ever: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ChestPain":
        data = data or {}
        quality = _to_str(data.get("quality"))
        if quality:
            aliases = {
                "squeezing": "pressure",
                "tight": "pressure",
                "tightness": "pressure",
                "crushing": "pressure",
                "heavy": "pressure",
                "heaviness": "pressure",
                "stabbing": "sharp",
                "ripping": "tearing",
                "ache": "dull",
                "aching": "dull",
            }
            quality = aliases.get(quality, quality)
            if quality not in PAIN_QUALITIES:
                quality = None
        location = _to_str(data.get("location"))
        if location:
            aliases = {
                "center": "substernal",
                "centre": "substernal",
                "central": "substernal",
                "middle": "substernal",
                "left_side": "left",
                "right_side": "right",
                "stomach": "epigastric",
                "upper_abdomen": "epigastric",
            }
            location = aliases.get(location.replace(" ", "_"), location)
            if location not in PAIN_LOCATIONS:
                location = None
        radiation = [
            r for r in _to_list(data.get("radiation")) if r in RADIATION_SITES
        ]
        severity = _to_int(data.get("severity"))
        if severity is not None:
            severity = max(0, min(10, severity))
        return cls(
            present=_to_bool(data.get("present")),
            quality=quality,
            location=location,
            radiation=radiation,
            ongoing=_to_bool(data.get("ongoing")),
            duration_minutes=_to_int(data.get("duration_minutes")),
            sudden_onset=_to_bool(data.get("sudden_onset")),
            exertional=_to_bool(data.get("exertional")),
            relieved_by_rest=_to_bool(data.get("relieved_by_rest")),
            relieved_by_nitroglycerin=_to_bool(data.get("relieved_by_nitroglycerin")),
            positional=_to_bool(data.get("positional")),
            pleuritic=_to_bool(data.get("pleuritic")),
            reproducible_on_palpation=_to_bool(data.get("reproducible_on_palpation")),
            severity=severity,
            worst_ever=_to_bool(data.get("worst_ever")),
        )


@dataclass
class Symptoms:
    dyspnea: Optional[bool] = None
    dyspnea_at_rest: Optional[bool] = None
    orthopnea: Optional[bool] = None
    paroxysmal_nocturnal_dyspnea: Optional[bool] = None
    diaphoresis: Optional[bool] = None
    nausea: Optional[bool] = None
    vomiting: Optional[bool] = None
    palpitations: Optional[bool] = None
    irregular_heartbeat: Optional[bool] = None
    syncope: Optional[bool] = None
    presyncope: Optional[bool] = None
    exertional_syncope: Optional[bool] = None
    dizziness: Optional[bool] = None
    leg_swelling: Optional[bool] = None
    unilateral_leg_swelling: Optional[bool] = None
    hemoptysis: Optional[bool] = None
    fever: Optional[bool] = None
    recent_viral_illness: Optional[bool] = None
    severe_headache: Optional[bool] = None
    vision_changes: Optional[bool] = None
    confusion: Optional[bool] = None
    focal_neuro_deficit: Optional[bool] = None
    fatigue: Optional[bool] = None
    rapid_weight_gain: Optional[bool] = None
    cough: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Symptoms":
        data = data or {}
        kwargs = {f.name: _to_bool(data.get(f.name)) for f in fields(cls)}
        # Common aliases from free-form extraction
        if kwargs["dyspnea"] is None:
            kwargs["dyspnea"] = _to_bool(data.get("shortness_of_breath") or data.get("sob"))
        if kwargs["diaphoresis"] is None:
            kwargs["diaphoresis"] = _to_bool(data.get("sweating"))
        if kwargs["syncope"] is None:
            kwargs["syncope"] = _to_bool(data.get("fainting") or data.get("passed_out"))
        if kwargs["presyncope"] is None:
            kwargs["presyncope"] = _to_bool(data.get("lightheadedness"))
        return cls(**kwargs)


@dataclass
class History:
    coronary_artery_disease: Optional[bool] = None
    prior_myocardial_infarction: Optional[bool] = None
    prior_stent_or_bypass: Optional[bool] = None
    heart_failure: Optional[bool] = None
    arrhythmia: Optional[bool] = None
    atrial_fibrillation: Optional[bool] = None
    hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    hyperlipidemia: Optional[bool] = None
    smoking: Optional[bool] = None
    family_history_premature_cad: Optional[bool] = None
    family_history_sudden_death: Optional[bool] = None
    prior_pe_or_dvt: Optional[bool] = None
    active_cancer: Optional[bool] = None
    recent_surgery_or_immobilization: Optional[bool] = None
    pregnancy_or_estrogen_use: Optional[bool] = None
    connective_tissue_disorder: Optional[bool] = None
    known_aortic_aneurysm: Optional[bool] = None
    chronic_kidney_disease: Optional[bool] = None
    stimulant_or_cocaine_use: Optional[bool] = None
    valvular_disease_or_murmur: Optional[bool] = None
    hypertrophic_cardiomyopathy: Optional[bool] = None
    pacemaker_or_icd: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "History":
        data = data or {}
        kwargs = {f.name: _to_bool(data.get(f.name)) for f in fields(cls)}
        aliases = {
            "coronary_artery_disease": ["cad", "heart_disease"],
            "prior_myocardial_infarction": ["prior_mi", "heart_attack"],
            "hyperlipidemia": ["high_cholesterol", "dyslipidemia", "dyslipidaemia"],
            "hypertension": ["high_blood_pressure"],
            "smoking": ["smoker", "tobacco"],
            "atrial_fibrillation": ["afib", "af"],
        }
        for key, alts in aliases.items():
            if kwargs[key] is None:
                for alt in alts:
                    if alt in data:
                        kwargs[key] = _to_bool(data.get(alt))
                        break
        return cls(**kwargs)


@dataclass
class ClinicalFindings:
    age: Optional[int] = None
    sex: Optional[str] = None  # "male" | "female"
    chief_complaint: Optional[str] = None
    chest_pain: ChestPain = field(default_factory=ChestPain)
    symptoms: Symptoms = field(default_factory=Symptoms)
    history: History = field(default_factory=History)
    vitals: Vitals = field(default_factory=Vitals)
    medications: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ClinicalFindings":
        """Build findings from loosely-typed JSON (LLM output or API body)."""
        data = data or {}
        sex = _to_str(data.get("sex") or data.get("gender"))
        if sex in {"m", "man", "male"}:
            sex = "male"
        elif sex in {"f", "woman", "female"}:
            sex = "female"
        else:
            sex = None

        # Accept a flat "legacy" shape (the existing Patient model) as well.
        chest_pain = data.get("chest_pain")
        if chest_pain is None and any(k in data for k in ("pain_quality", "location")):
            chest_pain = {
                "present": True,
                "quality": data.get("pain_quality"),
                "location": "substernal" if _to_bool(data.get("location")) else None,
                "exertional": _to_bool(data.get("trigger")),
                "relieved_by_rest": _to_bool(data.get("relief")),
            }
        symptoms = data.get("symptoms")
        if symptoms is None and any(k in data for k in ("sob", "shortness_of_breath")):
            symptoms = {"dyspnea": data.get("sob", data.get("shortness_of_breath"))}
        history = data.get("history")
        if history is None and any(
            k in data for k in ("hypertension", "diabetes", "hyperlipidemia", "smoking")
        ):
            history = {
                "hypertension": data.get("hypertension"),
                "diabetes": data.get("diabetes"),
                "hyperlipidemia": data.get("hyperlipidemia"),
                "smoking": data.get("smoking"),
            }

        return cls(
            age=_to_int(data.get("age")),
            sex=sex,
            chief_complaint=_to_str(data.get("chief_complaint")),
            chest_pain=ChestPain.from_dict(chest_pain),
            symptoms=Symptoms.from_dict(symptoms),
            history=History.from_dict(history),
            vitals=Vitals.from_dict(data.get("vitals")),
            medications=_to_list(data.get("medications")),
            notes=_to_str(data.get("notes")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Convenience accessors used throughout the pathway modules ---------- #
    @property
    def has_chest_pain(self) -> bool:
        return bool(self.chest_pain.present)

    @property
    def is_male(self) -> Optional[bool]:
        if self.sex is None:
            return None
        return self.sex == "male"

    def radiates_to(self, *sites: str) -> bool:
        return any(s in self.chest_pain.radiation for s in sites)

    def risk_factor_count(self) -> int:
        h = self.history
        return sum(
            1
            for v in (
                h.hypertension,
                h.diabetes,
                h.hyperlipidemia,
                h.smoking,
                h.family_history_premature_cad,
            )
            if v
        )

    def known_cad(self) -> bool:
        h = self.history
        return bool(
            h.coronary_artery_disease
            or h.prior_myocardial_infarction
            or h.prior_stent_or_bypass
        )
