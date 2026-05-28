import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def load_document(file_path: str):
    extension = os.path.splitext(file_path)[1].lower()
    loader_class = SUPPORTED_LOADERS.get(extension)

    if not loader_class:
        raise ValueError(f"Unsupported file type: {extension}")

    loader = loader_class(file_path)
    return loader.load()