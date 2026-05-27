import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================
# Configuration
# =========================


def get_embedding_model_name() -> str:
    model = os.getenv("GEMINI_EMBEDDINGS_MODEL")
    if not model:
        raise RuntimeError(
            "GEMINI_EMBEDDINGS_MODEL is not set. Add it to .env or export it in your environment."
        )
    return model


def get_gemini_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env or export it in your environment."
        )
    return api_key


# =========================
# Embedding Factory
# =========================

def create_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=get_embedding_model_name(),
        api_key=get_gemini_api_key(),
        output_dimensionality=1536
    )
