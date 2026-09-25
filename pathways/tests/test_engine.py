"""
Run from the backend directory:

    python3 -m unittest pathways.tests.test_engine -v

Pure stdlib — no FastAPI or database required.
"""

import unittest

from pathways.engine import evaluate, run_pathway
from pathways.extraction import extract_with_keywords, transcript_from_messages
from pathways.findings import ClinicalFindings


def findings(**kwargs):
    return ClinicalFindings.from_dict(kwargs)


class TestFindings(unittest.TestCase):
    def test_from_dict_is_tolerant(self):
        f = findings(
            age="58 years",
            gender="Male",
            chest_pain={"present": "yes", "quality": "Squeezing", "location": "Center", "radiation": "left arm, jaw", "severity": "8/10"},
            symptoms={"sweating": "Yes", "nausea": "no"},
            history={"high_blood_pressure": True, "diabetes": "unknown"},
        )
        self.assertEqual(f.age, 58)
        self.assertEqual(f.sex, "male")
        self.assertEqual(f.chest_pain.quality, "pressure")
        self.assertEqual(f.chest_pain.location, "substernal")
        self.assertIn("left_arm", f.chest_pain.radiation)
        self.assertEqual(f.chest_pain.severity, 8)
        self.assertTrue(f.symptoms.diaphoresis)
        self.assertFalse(f.symptoms.nausea)
        self.assertTrue(f.history.hypertension)
        self.assertIsNone(f.history.diabetes)

    def test_legacy_flat_patient_shape(self):
        f = findings(age=62, gender="female", pain_quality="pressure", location="Yes", sob="Yes", hypertension="Yes", diabetes="No", smoking="Yes")
        self.assertTrue(f.has_chest_pain)
        self.assertEqual(f.chest_pain.location, "substernal")
        self.assertTrue(f.symptoms.dyspnea)
        self.assertTrue(f.history.hypertension)
        self.assertFalse(f.history.diabetes)


class TestPathways(unittest.TestCase):
    def test_classic_acs_is_emergency(self):
        r = run_pathway(findings(
            age=63, sex="male",
            chest_pain={"present": True, "quality": "pressure", "location": "substernal", "radiation": ["left_arm", "jaw"], "ongoing": True, "duration_minutes": 40, "exertional": True, "severity": 8},
            symptoms={"diaphoresis": True, "nausea": True, "dyspnea": True},
            history={"hypertension": True, "diabetes": True, "smoking": True},
        ))
        self.assertEqual(r.primary.pathway, "acs")
        self.assertEqual(r.disposition.level.value, "emergency")
        self.assertTrue(any(f.code == "ongoing_ischemic_pain" for f in r.red_flags))
        self.assertEqual(r.next_questions, [])  # stop interviewing once it's an emergency
        self.assertGreaterEqual(r.risk_percent, 70)

    def test_stable_exertional_angina_is_urgent_or_prompt_not_emergency(self):
        r = run_pathway(findings(
            age=58, sex="male",
            chest_pain={"present": True, "quality": "pressure", "location": "substernal", "ongoing": False, "duration_minutes": 5, "exertional": True, "relieved_by_rest": True, "severity": 4},
            symptoms={"diaphoresis": False, "nausea": False, "dyspnea": False, "syncope": False},
            history={"hypertension": True, "diabetes": False, "hyperlipidemia": True, "smoking": False},
        ))
        self.assertEqual(r.primary.pathway, "acs")
        self.assertIn(r.disposition.level.value, {"urgent", "prompt"})
        cad = next(c for c in r.chronic if c["pathway"] == "cad_risk")
        self.assertEqual(cad["chest_pain_type"], "typical")
        self.assertIsNotNone(cad["probability"])

    def test_aortic_dissection(self):
        r = run_pathway(findings(
            age=66, sex="male",
            chest_pain={"present": True, "quality": "tearing", "sudden_onset": True, "worst_ever": True, "radiation": ["back"], "ongoing": True},
            history={"hypertension": True},
            vitals={"systolic_bp": 190, "diastolic_bp": 105},
        ))
        self.assertEqual(r.primary.pathway, "aortic")
        self.assertEqual(r.disposition.level.value, "emergency")
        self.assertTrue(any(f.code == "tearing_pain" for f in r.red_flags))

    def test_pulmonary_embolism(self):
        r = run_pathway(findings(
            age=34, sex="female",
            chest_pain={"present": True, "quality": "sharp", "pleuritic": True, "sudden_onset": True, "ongoing": True},
            symptoms={"dyspnea": True, "unilateral_leg_swelling": True, "hemoptysis": False},
            history={"recent_surgery_or_immobilization": True, "pregnancy_or_estrogen_use": True},
            vitals={"heart_rate": 112},
        ))
        self.assertEqual(r.primary.pathway, "pulmonary_embolism")
        self.assertIn(r.disposition.level.value, {"emergency", "urgent"})

    def test_hypertensive_emergency(self):
        r = run_pathway(findings(
            age=55, sex="female",
            symptoms={"severe_headache": True, "vision_changes": True},
            history={"hypertension": True},
            vitals={"systolic_bp": 205, "diastolic_bp": 125},
        ))
        self.assertEqual(r.primary.pathway, "hypertensive")
        self.assertEqual(r.disposition.level.value, "emergency")
        self.assertTrue(any(f.code == "hypertensive_emergency" for f in r.red_flags))

    def test_hypertensive_urgency_without_symptoms(self):
        r = run_pathway(findings(
            age=55, sex="female", history={"hypertension": True},
            vitals={"systolic_bp": 185, "diastolic_bp": 115},
            symptoms={"severe_headache": False, "vision_changes": False, "confusion": False},
        ))
        self.assertEqual(r.primary.pathway, "hypertensive")
        self.assertEqual(r.disposition.level.value, "urgent")
        htn = next(c for c in r.chronic if c["pathway"] == "hypertension_management")
        self.assertEqual(htn["bp_stage"], "crisis")

    def test_heart_failure(self):
        r = run_pathway(findings(
            age=74, sex="male",
            symptoms={"dyspnea": True, "orthopnea": True, "paroxysmal_nocturnal_dyspnea": True, "leg_swelling": True, "rapid_weight_gain": True, "dyspnea_at_rest": False},
            history={"heart_failure": True, "hypertension": True},
        ))
        self.assertEqual(r.primary.pathway, "heart_failure")
        self.assertEqual(r.primary.likelihood.value, "high")
        self.assertIn(r.disposition.level.value, {"urgent", "emergency"})

    def test_arrhythmia_unstable(self):
        r = run_pathway(findings(
            age=48, sex="female",
            symptoms={"palpitations": True, "irregular_heartbeat": True, "presyncope": True},
            history={"atrial_fibrillation": True},
            vitals={"heart_rate": 165},
        ))
        self.assertEqual(r.primary.pathway, "arrhythmia")
        self.assertEqual(r.disposition.level.value, "emergency")

    def test_arrhythmia_stable_palpitations(self):
        r = run_pathway(findings(
            age=29, sex="female",
            symptoms={"palpitations": True, "irregular_heartbeat": False, "presyncope": False, "syncope": False, "dyspnea": False},
        ))
        self.assertEqual(r.primary.pathway, "arrhythmia")
        self.assertIn(r.disposition.level.value, {"prompt", "routine"})

    def test_pericarditis(self):
        r = run_pathway(findings(
            age=27, sex="male",
            chest_pain={"present": True, "quality": "sharp", "positional": True, "pleuritic": True, "ongoing": True, "exertional": False, "duration_minutes": 600},
            symptoms={"recent_viral_illness": True, "fever": True, "diaphoresis": False, "nausea": False, "dyspnea": False, "syncope": False},
        ))
        self.assertEqual(r.primary.pathway, "pericarditis")
        self.assertIn(r.disposition.level.value, {"urgent", "prompt"})

    def test_exertional_syncope(self):
        r = run_pathway(findings(
            age=19, sex="male",
            symptoms={"syncope": True, "exertional_syncope": True, "palpitations": False},
            history={"family_history_sudden_death": True},
        ))
        self.assertEqual(r.primary.pathway, "syncope")
        self.assertEqual(r.disposition.level.value, "emergency")

    def test_benign_musculoskeletal_pain_is_prompt_with_questions(self):
        r = run_pathway(findings(
            age=24, sex="female",
            chest_pain={"present": True, "quality": "sharp", "location": "left", "reproducible_on_palpation": True, "exertional": False, "ongoing": False, "duration_minutes": 1, "sudden_onset": False, "worst_ever": False},
            symptoms={"dyspnea": False, "dyspnea_at_rest": False, "diaphoresis": False, "nausea": False, "syncope": False, "palpitations": False},
            history={"hypertension": False, "diabetes": False, "hyperlipidemia": False, "smoking": False},
        ))
        self.assertEqual(r.disposition.level.value, "prompt")
        self.assertEqual(r.red_flags, [])
        for a in r.assessments:
            self.assertNotEqual(a.likelihood.value, "high")

    def test_no_symptoms_is_routine(self):
        r = run_pathway(findings(age=40, sex="male", chest_pain={"present": False}))
        self.assertEqual(r.disposition.level.value, "routine")
        self.assertIsNotNone(r.disposition.patient_message)

    def test_next_questions_target_primary_pathway(self):
        r = run_pathway(findings(
            age=60, sex="male",
            chest_pain={"present": True, "quality": "pressure"},
        ))
        self.assertEqual(r.primary.pathway, "acs")
        self.assertTrue(r.next_questions)
        self.assertTrue(any("right now" in q for q in r.next_questions))

    def test_evaluate_returns_json_serialisable_dict(self):
        import json

        out = evaluate({"age": 50, "sex": "male", "chest_pain": {"present": True, "quality": "pressure", "exertional": True}})
        json.dumps(out)
        self.assertIn("disposition", out)
        self.assertIn("primary_pathway", out)
        self.assertIn("chronic_pathways", out)
        self.assertEqual(len(out["pathways"]), 8)


class TestKeywordExtraction(unittest.TestCase):
    def test_extracts_classic_story(self):
        messages = [
            {"role": "assistant", "content": "Can you describe your chest pain? Any sweating?"},
            {"role": "user", "content": "I'm 61 years old, male. I've had a crushing pressure in the center of my chest for 30 minutes, it goes down my left arm and I'm sweating a lot. I have high blood pressure and diabetes but I don't smoke. It's happening right now."},
        ]
        f = extract_with_keywords(transcript_from_messages(messages))
        self.assertEqual(f.age, 61)
        self.assertEqual(f.sex, "male")
        self.assertTrue(f.chest_pain.present)
        self.assertEqual(f.chest_pain.quality, "pressure")
        self.assertEqual(f.chest_pain.location, "substernal")
        self.assertIn("left_arm", f.chest_pain.radiation)
        self.assertEqual(f.chest_pain.duration_minutes, 30)
        self.assertTrue(f.chest_pain.ongoing)
        self.assertTrue(f.symptoms.diaphoresis)
        self.assertTrue(f.history.hypertension)
        self.assertTrue(f.history.diabetes)
        self.assertFalse(f.history.smoking)
        r = run_pathway(f)
        self.assertEqual(r.primary.pathway, "acs")
        self.assertEqual(r.disposition.level.value, "emergency")

    def test_assistant_questions_do_not_count_as_symptoms(self):
        messages = [
            {"role": "assistant", "content": "Do you have any sweating, nausea or shortness of breath?"},
            {"role": "user", "content": "No, none of those. Just a dull ache when I press on my chest."},
        ]
        f = extract_with_keywords(transcript_from_messages(messages))
        self.assertIsNone(f.symptoms.diaphoresis)
        self.assertEqual(f.chest_pain.quality, "dull")
        self.assertTrue(f.chest_pain.reproducible_on_palpation)

    def test_blood_pressure_parsing(self):
        f = extract_with_keywords("Patient: my blood pressure is high, it read 190 over 118 this morning and I have a bad headache.")
        self.assertEqual(f.vitals.systolic_bp, 190)
        self.assertEqual(f.vitals.diastolic_bp, 118)
        self.assertTrue(f.symptoms.severe_headache)
        r = run_pathway(f)
        self.assertEqual(r.disposition.level.value, "emergency")


if __name__ == "__main__":
    unittest.main()
