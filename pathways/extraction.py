"""
Symptom + history extraction.

Two extractors share one output schema (``ClinicalFindings``):

* ``extract_with_llm``     — structured JSON extraction via OpenAI (production).
* ``extract_with_keywords`` — deterministic regex fallback used when no API key
  is configured, in tests, and as a safety net if the model returns bad JSON.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Union

from .findings import ClinicalFindings

Message = Dict[str, str]

EXTRACTION_SCHEMA = {
    "age": "integer or null",
    "sex": "'male' | 'female' | null",
    "chief_complaint": "short string",
    "chest_pain": {
        "present": "bool|null",
        "quality": "'pressure'|'sharp'|'burning'|'tearing'|'dull'|'pleuritic'|null",
        "location": "'substernal'|'left'|'right'|'epigastric'|'diffuse'|null",
        "radiation": "list from ['arm','left_arm','both_arms','jaw','neck','back','shoulder']",
        "ongoing": "bool|null",
        "duration_minutes": "integer|null",
        "sudden_onset": "bool|null",
        "exertional": "bool|null",
        "relieved_by_rest": "bool|null",
        "relieved_by_nitroglycerin": "bool|null",
        "positional": "bool|null (worse lying flat / better leaning forward)",
        "pleuritic": "bool|null (worse with deep breath)",
        "reproducible_on_palpation": "bool|null",
        "severity": "0-10|null",
        "worst_ever": "bool|null",
    },
    "symptoms": {
        k: "bool|null"
        for k in (
            "dyspnea", "dyspnea_at_rest", "orthopnea", "paroxysmal_nocturnal_dyspnea",
            "diaphoresis", "nausea", "vomiting", "palpitations", "irregular_heartbeat",
            "syncope", "presyncope", "exertional_syncope", "dizziness", "leg_swelling",
            "unilateral_leg_swelling", "hemoptysis", "fever", "recent_viral_illness",
            "severe_headache", "vision_changes", "confusion", "focal_neuro_deficit",
            "fatigue", "rapid_weight_gain", "cough",
        )
    },
    "history": {
        k: "bool|null"
        for k in (
            "coronary_artery_disease", "prior_myocardial_infarction", "prior_stent_or_bypass",
            "heart_failure", "arrhythmia", "atrial_fibrillation", "hypertension", "diabetes",
            "hyperlipidemia", "smoking", "family_history_premature_cad",
            "family_history_sudden_death", "prior_pe_or_dvt", "active_cancer",
            "recent_surgery_or_immobilization", "pregnancy_or_estrogen_use",
            "connective_tissue_disorder", "known_aortic_aneurysm", "chronic_kidney_disease",
            "stimulant_or_cocaine_use", "valvular_disease_or_murmur",
            "hypertrophic_cardiomyopathy", "pacemaker_or_icd",
        )
    },
    "vitals": {
        "heart_rate": "int|null", "systolic_bp": "int|null", "diastolic_bp": "int|null",
        "spo2": "int|null", "respiratory_rate": "int|null", "temperature_c": "float|null",
    },
    "medications": "list of strings",
    "notes": "anything clinically relevant that does not fit above",
}

SYSTEM_PROMPT = (
    "You are a clinical information extraction system for a cardiovascular triage "
    "service. Read the conversation between a patient and an AI assistant and "
    "return ONLY a JSON object matching the schema. Use null for anything the "
    "patient was not asked or did not clearly state — never guess. Use true only "
    "when the patient affirms a symptom/history and false only when they deny it."
)


def transcript_from_messages(messages: Iterable[Union[Message, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role", "user"), m.get("content", "")
        else:  # SQLAlchemy Message objects
            role, content = getattr(m, "role", "user"), getattr(m, "content", "")
        speaker = "Assistant" if str(role).lower() in {"assistant", "agent"} else "Patient"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM extractor
# --------------------------------------------------------------------------- #

def extract_with_llm(transcript: str, client: Any, model: str = "gpt-4.1-mini") -> ClinicalFindings:
    """
    ``client`` is an ``openai.OpenAI`` instance. Falls back to the keyword
    extractor if the response is not valid JSON.
    """
    prompt = (
        "Schema:\n" + json.dumps(EXTRACTION_SCHEMA, indent=1)
        + "\n\nConversation:\n" + transcript
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return extract_with_keywords(transcript)
    findings = ClinicalFindings.from_dict(data)
    # Merge in anything the deterministic extractor is confident about that the
    # model left null (cheap belt-and-braces for safety-critical fields).
    fallback = extract_with_keywords(transcript)
    _fill_missing(findings, fallback)
    return findings


def _fill_missing(target: ClinicalFindings, source: ClinicalFindings) -> None:
    for section in ("chest_pain", "symptoms", "history"):
        t, s = getattr(target, section), getattr(source, section)
        for name in vars(t):
            if getattr(t, name) is None and getattr(s, name) is not None:
                setattr(t, name, getattr(s, name))
    if target.age is None:
        target.age = source.age
    if target.sex is None:
        target.sex = source.sex


# --------------------------------------------------------------------------- #
# Keyword extractor
# --------------------------------------------------------------------------- #

_NEG = r"(?:no|not|never|denies?|deny|without|haven't|hasn't|don't|doesn't|didn't|isn't|aren't)\s+(?:\w+\s+){0,3}"


def _mentions(text: str, pattern: str) -> Optional[bool]:
    """True if the pattern appears un-negated, False if only negated, None if absent."""
    positive = False
    negative = False
    for m in re.finditer(pattern, text):
        start = max(0, m.start() - 40)
        window = text[start : m.start()]
        if re.search(_NEG + r"$", window):
            negative = True
        else:
            positive = True
    if positive:
        return True
    if negative:
        return False
    return None


def extract_with_keywords(transcript: str) -> ClinicalFindings:
    """Only patient turns are analysed so the assistant's questions don't count as symptoms."""
    patient_text = " ".join(
        line.split(":", 1)[1] for line in transcript.splitlines() if line.lower().startswith("patient:")
    ) or transcript
    t = patient_text.lower()

    age = None
    m = re.search(r"\b(\d{1,3})\s*(?:years?\s*old|y/?o|yrs?)\b", t) or re.search(r"\bi(?:'m| am)\s+(\d{1,3})\b", t)
    if m:
        age = int(m.group(1))
    sex = "male" if re.search(r"\b(male|man|he/him)\b", t) and not re.search(r"\bfemale\b", t) else ("female" if re.search(r"\b(female|woman|she/her)\b", t) else None)

    cp = {
        "present": _mentions(
            t,
            r"chest[^.]{0,25}(?:pain|pressure|tight|discomfort|ache|hurt|squeez|crush|heav)"
            r"|(?:pain|pressure|tight\w*|discomfort|ache|squeez\w*|crush\w*|heav\w*)[^.]{0,40}chest",
        ),
        "quality": next(
            (q for pat, q in (
                (r"tearing|ripping", "tearing"),
                (r"pressure|squeez|tight|crush|heavy|elephant|band around", "pressure"),
                (r"sharp|stabbing|knife", "sharp"),
                (r"burning|heartburn", "burning"),
                (r"dull|ach(?:e|ing)", "dull"),
            ) if re.search(pat, t)),
            None,
        ),
        "location": "substernal" if re.search(r"cent(?:er|re) of (?:my )?chest|middle of (?:my )?chest|substernal|behind (?:my )?breastbone", t) else ("left" if re.search(r"left (?:side of my )?chest", t) else None),
        "radiation": [s for pat, s in (
            (r"left arm", "left_arm"), (r"both arms", "both_arms"), (r"\barm\b", "arm"),
            (r"\bjaw\b", "jaw"), (r"\bneck\b", "neck"), (r"\bback\b|shoulder blades", "back"), (r"shoulder", "shoulder"),
        ) if re.search(pat, t)],
        "ongoing": _mentions(t, r"right now|still (?:have|hurts|there)|currently|at the moment|hasn't stopped"),
        "sudden_onset": _mentions(t, r"sudden(?:ly)?|out of nowhere|all at once|within seconds"),
        "exertional": _mentions(t, r"(?:when|while|after) (?:i )?(?:walk|climb|exercis|exert|run|carry|stairs|uphill)|with (?:activity|exertion|exercise)"),
        "relieved_by_rest": _mentions(t, r"(?:better|goes away|eases?|stops?) (?:when|after|with|if) (?:i )?(?:rest|sit|stop)"),
        "relieved_by_nitroglycerin": _mentions(t, r"nitro"),
        "positional": _mentions(t, r"worse (?:when )?(?:lying|laying) (?:down|flat)|better (?:when )?(?:sitting|leaning) forward"),
        "pleuritic": _mentions(t, r"worse (?:when|with) (?:i )?(?:breath|deep breath|cough)"),
        "reproducible_on_palpation": _mentions(t, r"(?:hurts?|worse|tender|ache|pain\w*|sore) (?:when|if) (?:i )?(?:press|push|touch|poke)"),
        "worst_ever": _mentions(t, r"worst (?:pain )?(?:ever|of my life)"),
    }
    m = re.search(r"\b(\d{1,2})\s*(?:/|out of)\s*10\b", t)
    if m:
        cp["severity"] = int(m.group(1))
    m = re.search(r"(\d+)\s*(minute|min|hour|hr|day)s?\b", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        cp["duration_minutes"] = n * (1 if unit.startswith("min") else 60 if unit.startswith(("hour", "hr")) else 1440)

    symptoms = {
        "dyspnea": _mentions(t, r"short(?:ness)? of breath|breathless|can'?t (?:catch my )?breath|hard to breathe|trouble breathing"),
        "dyspnea_at_rest": _mentions(t, r"breath(?:less|e)?.{0,20}(?:at rest|sitting still|even resting)"),
        "orthopnea": _mentions(t, r"pillows|lying flat.{0,30}breath|prop(?:ped)? up"),
        "paroxysmal_nocturnal_dyspnea": _mentions(t, r"wake up.{0,30}(?:gasping|breath)"),
        "diaphoresis": _mentions(t, r"sweat|clammy|diaphore"),
        "nausea": _mentions(t, r"nause|queasy|sick to my stomach"),
        "vomiting": _mentions(t, r"vomit|threw up|throwing up"),
        "palpitations": _mentions(t, r"palpitation|racing|pounding|fluttering|heart (?:is )?(?:going )?fast"),
        "irregular_heartbeat": _mentions(t, r"irregular|skipping|skips? a beat|fluttering"),
        "syncope": _mentions(t, r"fainted|passed out|blacked out|lost consciousness|collapsed"),
        "presyncope": _mentions(t, r"nearly fainted|almost (?:fainted|passed out)|light-?headed"),
        "exertional_syncope": _mentions(t, r"(?:fainted|passed out|collapsed).{0,40}(?:exercis|running|exert|playing)"),
        "dizziness": _mentions(t, r"dizzy|dizziness"),
        "leg_swelling": _mentions(t, r"(?:legs?|ankles?|feet) (?:are |is )?swollen|swelling in my (?:legs?|ankles?)"),
        "unilateral_leg_swelling": _mentions(t, r"one (?:leg|calf)|(?:left|right) (?:leg|calf) (?:is )?(?:swollen|painful|sore)"),
        "hemoptysis": _mentions(t, r"cough(?:ed|ing)? (?:up )?blood|blood when i cough"),
        "fever": _mentions(t, r"fever|temperature|chills"),
        "recent_viral_illness": _mentions(t, r"(?:had|got over|recovering from) (?:a |the )?(?:cold|flu|virus|covid|viral)"),
        "severe_headache": _mentions(t, r"(?:severe|bad|worst|terrible) headache"),
        "vision_changes": _mentions(t, r"blurr|vision"),
        "confusion": _mentions(t, r"confus"),
        "focal_neuro_deficit": _mentions(t, r"weakness on one side|face (?:is )?droop|slurred|trouble speaking|numb(?:ness)? (?:in|on) (?:my )?(?:left|right|one)"),
        "fatigue": _mentions(t, r"fatigue|exhausted|tired all the time|no energy"),
        "rapid_weight_gain": _mentions(t, r"gained .{0,15}(?:pounds|kg|weight)"),
        "cough": _mentions(t, r"\bcough"),
    }

    history = {
        "coronary_artery_disease": _mentions(t, r"coronary|heart disease|blocked arter|angina"),
        "prior_myocardial_infarction": _mentions(t, r"heart attack|myocardial|\bmi\b"),
        "prior_stent_or_bypass": _mentions(t, r"stent|bypass|cabg"),
        "heart_failure": _mentions(t, r"heart failure|weak heart|chf"),
        "arrhythmia": _mentions(t, r"arrhythmia|irregular (?:heart )?rhythm"),
        "atrial_fibrillation": _mentions(t, r"atrial fib|a-?fib|afib"),
        "hypertension": _mentions(t, r"high blood pressure|hypertension|blood pressure (?:is |was )?high"),
        "diabetes": _mentions(t, r"diabet"),
        "hyperlipidemia": _mentions(t, r"cholesterol|hyperlipid|statin"),
        "smoking": _mentions(t, r"\bsmok|cigarette|vape"),
        "family_history_premature_cad": _mentions(t, r"(?:father|mother|brother|sister|dad|mom|parent).{0,40}heart (?:attack|disease)"),
        "family_history_sudden_death": _mentions(t, r"(?:died|passed) suddenly|sudden (?:cardiac )?death"),
        "prior_pe_or_dvt": _mentions(t, r"blood clot|\bdvt\b|pulmonary embol|\bpe\b"),
        "active_cancer": _mentions(t, r"cancer|chemo"),
        "recent_surgery_or_immobilization": _mentions(t, r"surgery|operation|long (?:flight|drive)|bed ?rest|cast on"),
        "pregnancy_or_estrogen_use": _mentions(t, r"pregnan|birth control|the pill|estrogen|hormone"),
        "connective_tissue_disorder": _mentions(t, r"marfan|ehlers|connective tissue"),
        "known_aortic_aneurysm": _mentions(t, r"aneurysm"),
        "chronic_kidney_disease": _mentions(t, r"kidney disease|dialysis|ckd"),
        "stimulant_or_cocaine_use": _mentions(t, r"cocaine|meth|amphetamine|stimulant|energy drinks"),
        "valvular_disease_or_murmur": _mentions(t, r"murmur|valve"),
        "hypertrophic_cardiomyopathy": _mentions(t, r"hypertrophic|hcm|cardiomyopathy"),
        "pacemaker_or_icd": _mentions(t, r"pacemaker|defibrillator|icd"),
    }

    vitals: Dict[str, Any] = {}
    m = re.search(r"\b(\d{2,3})\s*(?:/|over)\s*(\d{2,3})\b", t)
    if m and 70 <= int(m.group(1)) <= 260:
        vitals["systolic_bp"], vitals["diastolic_bp"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(?:heart rate|pulse)(?: is| was| of)?\s*(?:about |around )?(\d{2,3})", t) or re.search(r"\b(\d{2,3})\s*(?:bpm|beats)", t)
    if m:
        vitals["heart_rate"] = int(m.group(1))
    m = re.search(r"(?:oxygen|sat(?:uration)?|spo2)\D{0,12}(\d{2})\s*%?", t)
    if m:
        vitals["spo2"] = int(m.group(1))

    return ClinicalFindings.from_dict(
        {
            "age": age,
            "sex": sex,
            "chest_pain": cp,
            "symptoms": symptoms,
            "history": history,
            "vitals": vitals,
        }
    )
