from app.retrieval.retrieval_credit import query_credit_document


def normalize_query(question):
    q = question.lower()
    expansions = {
        "dti": "debt to income ratio",
        "emi": "equated monthly installment",
        "ltv": "loan to value ratio",
        "cibil": "credit score",
    }

    for k, v in expansions.items():
        if k in q:
            q += f" {v}"

    return q


def retrieve_context(question):
    query = normalize_query(question)
    docs = query_credit_document(query, k=5)
    context = []

    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata", {})
        context.append(
            f"Source {i}\n\n"
            f"File:\n{meta.get('source')}\n\n"
            f"Page:\n{meta.get('page')}\n\n"
            f"Content:\n{doc.get('content')}\n"
        )

    return docs, "\n".join(context)