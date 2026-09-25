# Clinical pathway engine

Deterministic, rules-based cardiovascular triage that sits between the AI
conversation (text or voice) and the recommendation the patient receives.

```
Patient
  ↓
Voice / Text                       agent.py (LiveKit) · main.py /api/chat/*
  ↓
Symptom + History Extraction       pathways/extraction.py  → ClinicalFindings
  ↓
Safety / Red-Flag Layer            pathways/red_flags.py
  ↓
Cardiovascular Triage Router       pathways/router.py
  ├── Acute coronary syndrome / ischemia   conditions/acs.py
  ├── Arrhythmia                           conditions/arrhythmia.py
  ├── Heart failure                        conditions/heart_failure.py
  ├── Hypertensive emergency               conditions/hypertensive.py
  ├── Pulmonary embolism                   conditions/pulmonary_embolism.py
  ├── Aortic emergency                     conditions/aortic.py
  ├── Pericarditis / myocarditis           conditions/pericarditis.py
  └── Syncope / structural disease         conditions/syncope.py
  ↓
Condition-specific assessment      each module scores evidence, lists what's missing
  ↓
Disposition recommendation         pathways/disposition.py
  Emergency · Urgent · Prompt follow-up (24–72 h) · Routine scheduling
```

Chronic pathways run alongside and enrich the result without competing for
"primary": **CAD risk** (`conditions/cad_risk.py`, wraps the CAD Consortium
model in `calculate_cad_score.py`), **hypertension** staging and
**diabetes** cardiometabolic risk.

## Using it

```python
from pathways import run_pathway, ClinicalFindings

result = run_pathway(ClinicalFindings.from_dict({
    "age": 63, "sex": "male",
    "chest_pain": {"present": True, "quality": "pressure", "radiation": ["left_arm"], "ongoing": True, "duration_minutes": 40},
    "symptoms": {"diaphoresis": True},
    "history": {"hypertension": True, "diabetes": True},
}))
result.disposition.level        # Disposition.EMERGENCY
result.primary.pathway          # "acs"
result.red_flags                # [RedFlag(code="ongoing_ischemic_pain", ...)]
result.next_questions           # [] — interviewing stops once it's an emergency
result.to_dict()                # JSON for the API / dashboards
```

HTTP (mounted in `main.py`):

| Method | Path                          | Purpose                                            |
| ------ | ----------------------------- | -------------------------------------------------- |
| POST   | `/api/pathway/evaluate`       | `{findings}` → full result                         |
| POST   | `/api/pathway/from-transcript`| `{messages|transcript}` → extraction → result      |
| GET    | `/api/pathway/catalog`        | pathway / disposition definitions for the UI       |

`/api/chat/message` now runs the red-flag layer on **every** turn (keyword
extraction, no LLM cost) and stops the interview with a 911 instruction when
an emergency flag fires. On completion it runs full LLM extraction, stores a
`PathwayEvaluation` row and returns `pathway` alongside `messages`.

## Tri-state findings

Every boolean in `ClinicalFindings` is `True` / `False` / `None`. `None` means
"not asked". The condition modules turn `None` into the next question the
agent should ask, so the interview adapts to the leading pathway.

## Tests

```bash
cd backend
python3 -m unittest pathways.tests.test_engine -v
```

Pure stdlib — no FastAPI, OpenAI key or database needed.

## Clinical basis (heuristics, not a validated model)

* ACS weights follow the HEART score components and the typical-angina triad.
* PE features follow the Wells criteria.
* Hypertension staging follows ACC/AHA 2017; ≥180/120 with organ symptoms is an emergency.
* Aortic dissection is escalated at "possible" because of its lethality.
* The engine is a triage aid; anything "likely" or above needs ECG/troponin in person.
