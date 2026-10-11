
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    """Return the Ollama embedding object used throughout the RAG project."""
    return OllamaEmbeddings(model="nomic-embed-text")
