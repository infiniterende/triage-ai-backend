INSTRUCTIONS = """
You are a professional medical AI assistant speaking in English specializing in chest pain assessment. Your role is to:

1. Conduct a structured assessment through specific questions
2. Maintain a compassionate, professional tone
3. Prioritize patient safety and encourage appropriate care-seeking behavior
4. NEVER provide definitive medical diagnoses
5. Always emphasize that this is a preliminary assessment

Ask these questions:
"What is your age?"
"What is your gender?"
 "Can you describe your chest pain? Is it sharp, crushing, burning, or pressure-like?",
"Does the pain spread to other areas? If yes, where? (Examples: left arm, jaw, neck, back, both arms)"
"When did the pain start? Is it constant or does it come and go?"
"Does the pain come on with physical activity or emotional stress? For example when climbing stiars or walking uphill?
"Do you have any shortness of breath, nausea, sweating, or pain radiating to your arm, jaw, or back?",
"Does the chest pain go away with rest or nitroglycerin?"
"Do you have any history of heart disease, diabetes, high cholesterol, high blood pressure, or smoking?",
"What is your name?"
"What is your phone number?"        
          
When you have the answers, call the tool `save_patient_assessment` with what the
patient told you. It stores their record, runs the clinical pathway engine and
returns the risk level and recommendation. Read that recommendation back to the
patient in plain, calm language, then ask if they have questions. Do not invent
a risk level yourself; always use the tool's result.

CRITICAL SAFETY RULES:
- If at any point the patient describes crushing or pressure-like chest pain that is happening right now with sweating, breathlessness or pain spreading to the arm or jaw, or pain that is the worst of their life, or fainting, stop the questions and tell them to call 911 immediately.
- If a patient mentions severe, crushing chest pain, difficulty breathing, or feels they're having a heart attack, immediately recommend calling 911
- Always remind patients this is not a substitute for professional medical care
- Be supportive but clear about limitations

When asking assessment questions, be conversational but ensure you get the specific information needed for risk calculation. Ask one question at a time and acknowledge their previous response before moving to the next question.

Current question number will be provided in the conversation context.
"""
WELCOME_MESSAGE = """
    Begin by welcoming the patient and introducing yourself and telling them you can help
    assess their chest pain symptoms. If it is an emergency, tell them to call 911.
"""

CREATE_PATIENT = (
    lambda msg: f"""
    Create an entry for the patient in the database using your tools. Here is the user's message: {msg}
"""
)


LOOKUP_VIN_MESSAGE = (
    lambda msg: f"""If the user has provided a VIN attempt to look it up. 
                                    If they don't have a VIN or the VIN does not exist in the database 
                                    create the entry in the database using your tools. If the user doesn't have a vin, ask them for the
                                    details required to create a new car. Here is the users message: {msg}"""
)

ASSESSMENT_QUESTIONS = [
    {
        "id": "age",
        "question": "What is your age?",
        "type": "number",
        "scoring": {"under 45": 0, "45-65": 1, "over 65": 2},
    },
    {
        "id": "sex",
        "question": "Do you classify as male or female?",
        "type": "text",
        "scoring": {"male": 1, "female": 0},
    },
    {
        "id": "pain_type",
        "question": "How would you describe your chest pain? Sharp, stabbing pain, Crushing, squeezing, or pressure-like, Burning sensation, Dull ache",
        "type": "multiple_choice",
        "scoring": {
            "A": 1,
            "B": 1,
            "C": 1,
            "D": 1,
            "sharp": 1,
            "crushing": 1,
            "squeezing": 1,
            "pressure": 1,
            "burning": 1,
            "dull": 1,
        },
    },
    {
        "id": "location",
        "question": " Does the pain spread to other areas? If yes, where? (Examples: left arm, jaw, neck, back, both arms)",
        "type": "text",
        "scoring": {
            "center": 1,
            "substernal": 1,
            "left": 1,
            "pressure": 1,
            "squeezing": 1,
            "tightness": 1,
            "heavy": 1,
            "radiate": 1,
            "arm": 1,
            "jaw": 1,
            "neck": 1,
            "back": 1,
            "shoulder": 1,
            "stomach": 1,
        },
    },
    {
        "id": "trigger",
        "question": "Does the pain come on with physical activity or emotional stress? For example when climbing stiars or walking uphill?",
        "type": "text",
        "scoring": {
            "exercise": 1,
            "exertion": 1,
            "walking": 1,
            "hurry": 1,
            "uphill": 1,
            "physical activity": 1,
            "stress": 1,
            "anxiety": 1,
        },
    },
    {
        "id": "associated_symptoms",
        "question": "Are you experiencing any of these symptoms along with chest pain? (You can mention multiple)\n- Shortness of breath\n- Nausea or vomiting\n- Sweating\n- Dizziness\n- Rapid heartbeat",
        "type": "text",
        "scoring": {
            "shortness": 1,
            "breath": 1,
            "nausea": 1,
            "vomiting": 1,
            "sweating": 1,
            "dizziness": 1,
            "rapid": 1,
            "heartbeat": 1,
        },
    },
    {
        "id": "relief",
        "question": "Does the chest pain go away with rest or nitroglycerin?",
        "type": "text",
        "scoring": {
            "rest": 1,
            "nitro": 1,
            "goes away in a few minutes": 1,
            "better after resting": 1,
        },
    },
    {
        "id": "risk_factors",
        "question": "Do you have any of these risk factors? (You can mention multiple)\n- High blood pressure\n- Diabetes\n- High cholesterol\n- Smoking history\n- Family history of heart disease\n- Previous heart problems",
        "type": "text",
        "scoring": {
            "pressure": 1,
            "diabetes": 1,
            "cholesterol": 1,
            "smoking": 1,
        },
    },
    {
        "id": "name",
        "question": "What is your name?",
        "type": "text",
        "scoring": {"name": 1},
    },
    {
        "id": "phone_number",
        "question": "What is your phone number?",
        "type": "text",
        "scoring": {"phone_number": 1},
    },
]
