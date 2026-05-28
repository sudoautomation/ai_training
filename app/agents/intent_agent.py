def detect_intent(question):
    q = question.lower()

    assessment = ["borrower", "approve", "eligible", "risk", "profile", "credit"]
    calculation = ["emi", "dti", "calculate", "interest"]
    policy = ["rbi", "policy", "guideline", "how", "why"]

    if any(x in q for x in assessment):
        return "assessment"

    if any(x in q for x in calculation):
        return "calculation"

    if any(x in q for x in policy):
        return "policy"

    return "unknown"