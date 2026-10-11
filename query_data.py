
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function


# Use absolute paths so the app always finds the correct database.
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma"

COLLECTION_NAME = "rag_documents"
LLM_MODEL = "llama3.2"
TOP_K = 3


PROMPT_TEMPLATE = """
You are a helpful assistant answering questions about PDF documents.

Use only the information provided in the context.
If the answer is not in the context, say:
"I could not find this information in the uploaded documents."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""


@st.cache_resource
def load_chroma_db():
    # Always create and pass the actual embedding object.
    embedding_model = get_embedding_function()

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


@st.cache_resource
def load_model():
    return OllamaLLM(
        model=LLM_MODEL,
        num_predict=800,
    )


def query_rag(query_text: str):
    """Retrieve relevant PDF chunks and generate an answer."""
    if not CHROMA_PATH.exists():
        return (
            "The database does not exist yet. Add PDFs to the data folder "
            "and run populate_database.py first.",
            [],
        )

    db = load_chroma_db()

    # Check whether any documents are indexed.
    stored_data = db.get()

    if not stored_data.get("ids"):
        return (
            "The database is empty. Add PDFs to the data folder and run "
            "python populate_database.py --reset first.",
            [],
        )

    # Retrieve the most relevant chunks.
    results = db.similarity_search(query_text, k=TOP_K)

    if not results:
        return (
            "I could not find relevant information in the uploaded documents.",
            [],
        )

    context_parts = []
    sources = []

    for document in results:
        source = Path(
            str(document.metadata.get("source", "Unknown"))
        ).name

        page = document.metadata.get("page")

        if page is not None:
            source_label = f"{source}, page {int(page) + 1}"
        else:
            source_label = source

        context_parts.append(
            f"Source: {source_label}\n"
            f"Content: {document.page_content}"
        )

        if source_label not in sources:
            sources.append(source_label)

    context_text = "\n\n---\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format(
        context=context_text,
        question=query_text,
    )

    model = load_model()
    answer = model.invoke(prompt)

    return answer, sources


def main():
    st.set_page_config(
        page_title="Local PDF RAG Chatbot",
        page_icon="📚",
        layout="centered",
    )

    st.title("📚 RAG Chatbot")
    st.write("Ask questions and get responses based on your uploaded PDFs.")

    with st.sidebar:
        st.header("Configuration")
        st.write(f"**Language model:** {LLM_MODEL}")
        st.write("**Embedding model:** nomic-embed-text")
        st.write(f"**Retrieved chunks:** {TOP_K}")
        st.caption("Your models run locally through Ollama.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.write(f"- {source}")

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching PDFs and generating an answer..."):
                try:
                    answer, sources = query_rag(question)

                    st.markdown(answer)

                    if sources:
                        with st.expander("View sources"):
                            for source in sources:
                                st.write(f"- {source}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    st.error(f"An error occurred: {error}")
                    st.info(
                        "Check the embedding configuration, Ollama server, "
                        "and Chroma database."
                    )


if __name__ == "__main__":
    main()
