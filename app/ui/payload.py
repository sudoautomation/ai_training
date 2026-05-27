def build_payload(query):

    payload = {

        "question":
        query
    }

    profile = getattr(
        __import__("streamlit")
        .session_state,

        "borrower_profile",

        None
    )

    if profile:

        payload[
            "borrower_profile"
        ] = profile

    return payload