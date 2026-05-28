def calculate_dti(profile):
    income = profile.get("monthly_income", 0)
    emis = profile.get("existing_emis", 0)
    loan = profile.get("requested_loan_amount", 0)
    tenure = profile.get("tenure_months", 0)

    if income == 0:
        return {}

    emi = 0

    if loan and tenure:
        r = 0.085 / 12
        emi = (loan * r * (1 + r) ** tenure) / ((1 + r) ** tenure - 1)

    dti = ((emis + emi) / income) * 100

    return {
        "emi": round(emi, 2),
        "dti": round(dti, 2)
    }