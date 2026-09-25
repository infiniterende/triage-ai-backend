import math


def classify_chest_pain(location: bool, trigger: bool, relief: bool):
    score = sum([location or 0, trigger or 0, relief or 0])
    if score == 3:
        return "typical"
    elif score == 2:
        return "atypical"
    else:
        return "non-specific"


def cadc_clinical_risk(
    age,
    male,
    chest_pain_type,
    diabetes=False,
    hypertension=False,
    dyslipidaemia=False,
    smoking=False,
):
    # Coefficients from BMJ 2012 Appendix Table 4
    b0 = -7.539
    b_age = 0.062
    b_male = 1.332
    b_atyp = 0.633
    b_typ = 1.998
    b_diab = 0.828
    b_htn = 0.338
    b_dlp = 0.422
    b_smk = 0.461
    b_interact = -0.402

    male = 1 if male else 0
    cp_atyp = 1 if chest_pain_type == "atypical" else 0
    cp_typ = 1 if chest_pain_type == "typical" else 0
    dm = 1 if diabetes else 0
    htn = 1 if hypertension else 0
    dlp = 1 if dyslipidaemia else 0
    smk = 1 if smoking else 0

    logit_p = (
        b0
        + b_age * age
        + b_male * male
        + b_atyp * cp_atyp
        + b_typ * cp_typ
        + b_diab * dm
        + b_htn * htn
        + b_dlp * dlp
        + b_smk * smk
        + b_interact * (dm * cp_typ)
    )
    return 1 / (1 + math.exp(-logit_p))


if __name__ == "__main__":
    # Example voice agent flow:
    # Collect answers from questions
    location = True  # Patient says substernal pressure spreading to left arm
    trigger = True  # Happens on exertion
    relief = False  # Does NOT always go away with rest

    # Classify chest pain type
    chest_pain = classify_chest_pain(location, trigger, relief)

    # Calculate risk
    prob = cadc_clinical_risk(
        age=55,
        male=True,
        chest_pain_type=chest_pain,
        diabetes=True,
        hypertension=True,
        dyslipidaemia=False,
        smoking=True,
    )

    print(f"Chest pain type: {chest_pain}")
    print(f"Predicted probability of CAD: {prob:.2%}")
