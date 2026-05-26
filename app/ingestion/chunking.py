from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs,
    chunk_size: int = 20000,
    chunk_overlap: int = 200
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(docs)
