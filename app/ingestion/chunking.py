from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs, chunk_size: int = 512, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # paragraph
            "\n",    # line
            ".",     # sentence
            " ",     # word
            ""       # character fallback
        ]
    )

    return splitter.split_documents(docs)