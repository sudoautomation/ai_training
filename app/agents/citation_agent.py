def build_citations(docs):

    citations=[]

    for doc in docs:

        meta=doc.get(
            "metadata",
            {}
        )

        citations.append({

            "title":

            f"{meta.get('source')} "

            f"Page {meta.get('page')}",

            "content":

            doc.get(
                "content",
                ""
            )[:300]
        })

    return citations