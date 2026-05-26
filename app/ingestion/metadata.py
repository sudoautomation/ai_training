import os


def enrich_metadata(docs, file_path: str):

    last_updated = os.path.getmtime(file_path)

    for doc in docs:
        doc.metadata.update({
            "source": file_path,
            "document_extension": "pdf",
            "page": doc.metadata.get("page"),
            "last_updated": last_updated
        })

    return docs