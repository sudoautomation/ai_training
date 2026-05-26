import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================
# Configuration
# =========================


def get_embedding_model_name() -> str:
    return os.getenv("GOOGLE_EMBEDDINGS_MODEL")


def get_gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY")


# =========================
# Embedding Factory
# =========================

def create_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=get_embedding_model_name(),
        api_key=get_gemini_api_key(),
        output_dimensionality=1536
    )
