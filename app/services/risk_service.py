def evaluate_risk(
    profile,
    dti
):

    score = 10

    if profile.get(
        "credit_score",
        0
    ) < 650:

        score -= 3

    if profile.get(
        "past_defaults",
        0
    ):

        score -= 2


    if dti.get(
        "dti",
        100
    ) > 50:

        score -= 3

    score=max(score,1)

    return {

        "risk_score":
        score,

        "risk_level":
        "LOW"
        if score>=8
        else "MEDIUM"
        if score>=5
        else "HIGH",

        "recommendation":
        "APPROVE"
        if score>=8
        else "CONDITIONAL APPROVE"
        if score>=5
        else "REJECT"
    }