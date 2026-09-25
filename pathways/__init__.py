"""
Agilance clinical pathway engine.

Pipeline (see README.md):

    Patient → Voice / Text
        → Symptom + History Extraction        (extraction.py → ClinicalFindings)
        → Safety / Red-Flag Layer             (red_flags.py)
        → Cardiovascular Triage Router        (router.py)
            ├── Acute coronary syndrome / ischemia
            ├── Arrhythmia
            ├── Heart failure
            ├── Hypertensive emergency
            ├── Pulmonary embolism
            ├── Aortic emergency
            ├── Pericarditis / myocarditis
            └── Syncope / structural disease
        → Condition-specific assessment       (conditions/*.py)
        → Disposition recommendation          (disposition.py)
            Emergency / Urgent / Prompt follow-up / Routine scheduling

`engine.run_pathway()` is the single entry point; `api.py` exposes it over HTTP.
"""

from .engine import run_pathway, PathwayResult
from .findings import ClinicalFindings

__all__ = ["run_pathway", "PathwayResult", "ClinicalFindings"]
