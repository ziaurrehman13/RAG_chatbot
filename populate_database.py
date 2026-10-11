
import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from get_embedding_function import get_embedding_function


# Project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma"

COLLECTION_NAME = "rag_documents"


def load_documents():
    """Load all PDF files from the data folder."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    return loader.load()


def split_documents(documents: list[Document]):
    """Split PDF text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )

    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks: list[Document]):
    """Create a unique ID for each text chunk."""
    page_counters = {}

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page", 0)

        page_id = f"{source}:{page}"
        chunk_index = page_counters.get(page_id, 0)

        chunk.metadata["id"] = f"{page_id}:{chunk_index}"
        page_counters[page_id] = chunk_index + 1

    return chunks


def clear_database():
    """Delete the existing Chroma database."""
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        print("Existing database deleted.")


def add_to_chroma(chunks: list[Document]):
    """Add document chunks using the same Ollama embeddings as querying."""
    if not chunks:
        print("No text chunks found.")
        return

    chunks = calculate_chunk_ids(chunks)

    # IMPORTANT: Pass the embedding OBJECT, not the function itself.
    embedding_model = get_embedding_function()

    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    existing_data = db.get()
    existing_ids = set(existing_data["ids"])

    new_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if not new_chunks:
        print("No new chunks to add.")
        return

    new_ids = [chunk.metadata["id"] for chunk in new_chunks]

    print(f"Adding {len(new_chunks)} chunks to the database...")
    db.add_documents(documents=new_chunks, ids=new_ids)

    print("PDF chunks and embeddings saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Create a local RAG database from PDF files."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the old database before indexing PDFs.",
    )
    args = parser.parse_args()

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if args.reset:
        print("Resetting database...")
        clear_database()

    print(f"Reading PDFs from: {DATA_PATH}")
    documents = load_documents()

    if not documents:
        print("No PDF pages found.")
        print("Place your PDF files inside the project's data folder.")
        return

    print(f"Loaded {len(documents)} PDF pages.")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    add_to_chroma(chunks)
    print("Database preparation finished.")


if __name__ == "__main__":
    main()
