import streamlit as st


def render_risk(risk):

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Score",
        risk["risk_score"]
    )

    c2.metric(
        "Threat Level",
        risk["risk_level"]
    )

    c3.metric(
        "Recommendation",
        risk["recommendation"]
    )