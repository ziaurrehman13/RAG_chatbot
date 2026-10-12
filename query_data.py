
import hashlib
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personal PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

MEMORY_TOP_K = 5
MEMORY_THRESHOLD = 0.25
VECTOR_DIMENSIONS = 384

# OCR settings
OCR_MIN_TEXT_CHARS = 30
OCR_RENDER_SCALE = 2.0


# ============================================================
# 2. PASSWORD LOGIN
# ============================================================

def require_login():
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 Personal RAG Chatbot")
    st.write("Enter your private password to open your chatbot.")

    with st.form("login_form"):
        password = st.text_input(
            "App password",
            type="password",
        )
        submitted = st.form_submit_button("Sign in")

    if submitted:
        expected = st.secrets.get("APP_PASSWORD", "")

        if expected and password == expected:
            st.session_state["authenticated"] = True
            st.rerun()

        elif not expected:
            st.error(
                "APP_PASSWORD is missing from Streamlit Cloud Secrets."
            )

        else:
            st.error("Incorrect password.")

    return False


if not require_login():
    st.stop()


# ============================================================
# 3. SUPABASE CHAT HISTORY
# ============================================================

@st.cache_resource
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SECRET_KEY"],
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_saved_chats():
    response = (
        get_supabase()
        .table("rag_chats")
        .select("*")
        .order("updated_at", desc=True)
        .execute()
    )

    chats = {}

    for row in response.data:
        chats[row["id"]] = {
            "title": row.get("title") or "New chat",
            "messages": row.get("messages") or [],
            "updated_at": row.get("updated_at") or now_iso(),
            "upload_signature": None,
            "upload_chunks": [],
            "upload_vectors": np.empty(
                (0, VECTOR_DIMENSIONS),
                dtype=np.float32,
            ),
        }

    return chats


def save_chat(chat_id, chat):
    chat["updated_at"] = now_iso()

    get_supabase().table("rag_chats").upsert({
        "id": chat_id,
        "title": chat["title"],
        "messages": chat["messages"],
        "updated_at": chat["updated_at"],
    }).execute()


def create_new_chat():
    chat_id = str(uuid.uuid4())

    chat = {
        "title": "New chat",
        "messages": [],
        "updated_at": now_iso(),
        "upload_signature": None,
        "upload_chunks": [],
        "upload_vectors": np.empty(
            (0, VECTOR_DIMENSIONS),
            dtype=np.float32,
        ),
    }

    st.session_state.chats[chat_id] = chat
    st.session_state.active_chat = chat_id

    save_chat(chat_id, chat)


# ============================================================
# 4. EMBEDDING MODEL AND GROQ
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    return Groq(api_key=api_key)


@st.cache_data(show_spinner=False)
def cached_embeddings(texts):
    if not texts:
        return np.empty(
            (0, VECTOR_DIMENSIONS),
            dtype=np.float32,
        )

    vectors = load_embedding_model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=16,
    )

    return np.asarray(vectors, dtype=np.float32)


# ============================================================
# 5. TEXT CHUNKING
# ============================================================

def split_text(text):
    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        part = text[start:end].strip()

        if part:
            chunks.append(part)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# 6. OCR HELPERS
# ============================================================

def perform_ocr(page):
    """
    Render one PDF page as an image and extract its text using OCR.

    Requires PyMuPDF, Pillow, pytesseract, and the Tesseract
    OCR engine installed on the computer or hosting platform.
    """

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(
            OCR_RENDER_SCALE,
            OCR_RENDER_SCALE,
        ),
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # Basic image preprocessing to improve OCR readability.
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)

    text = pytesseract.image_to_string(
        image,
        config="--psm 3",
    )

    return text.strip()


def extract_pdf_pages(pdf_bytes, filename):
    """
    Extract text page by page.

    1. Attempt native PDF text extraction.
    2. If a page has very little text, attempt OCR.
    3. Preserve the original filename and page number.
    """

    extracted_pages = []

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:

        for page_number, page in enumerate(pdf, start=1):
            # First attempt normal PDF text extraction.
            try:
                native_text = page.get_text("text").strip()
            except Exception as error:
                print(
                    f"Native extraction failed for {filename}, "
                    f"page {page_number}: {error}"
                )
                native_text = ""

            page_text = native_text
            extraction_method = "native"

            # Scanned/image-only pages usually have little or no
            # extractable text, so attempt OCR on those pages.
            if len(native_text) < OCR_MIN_TEXT_CHARS:
                try:
                    ocr_text = perform_ocr(page)

                    # Use OCR output when it provides more text.
                    if len(ocr_text) > len(native_text):
                        page_text = ocr_text
                        extraction_method = "OCR"

                except Exception as error:
                    # Keep the application running if OCR fails.
                    print(
                        f"OCR failed for {filename}, "
                        f"page {page_number}: {error}"
                    )

            extracted_pages.append({
                "source": filename,
                "page": page_number,
                "text": page_text,
                "method": extraction_method,
            })

    return extracted_pages


def pages_to_documents(extracted_pages):
    """
    Convert extracted page text into the existing RAG chunks.
    Source filenames and page numbers remain attached to each chunk.
    """

    documents = []

    for item in extracted_pages:
        page_text = item["text"]

        if not page_text.strip():
            continue

        for part in split_text(page_text):
            documents.append({
                "text": part,
                "source": item["source"],
                "page": item["page"],
            })

    return documents


def read_pdf_bytes(pdf_bytes, filename):
    """
    Shared PDF reader for repository PDFs and uploaded PDFs.
    """

    extracted_pages = extract_pdf_pages(
        pdf_bytes,
        filename,
    )

    return pages_to_documents(extracted_pages)


# ============================================================
# 7. PROJECT FOLDER PDF LOADING
# ============================================================

def get_folder_signature():
    # Never create directories in the deployed repository.
    if not DATA_DIR.is_dir():
        return tuple()

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    return tuple(
        (
            file.name,
            file.stat().st_size,
            file.stat().st_mtime_ns,
        )
        for file in pdf_files
    )


@st.cache_data(show_spinner=False)
def load_folder_pdfs(folder_path, signature):
    documents = []

    for filename, _, _ in signature:
        pdf_path = Path(folder_path) / filename

        try:
            pdf_bytes = pdf_path.read_bytes()

            documents.extend(
                read_pdf_bytes(
                    pdf_bytes,
                    filename,
                )
            )

        except Exception as error:
            print(
                f"Could not read or OCR {filename}: {error}"
            )

    return documents


# ============================================================
# 8. UPLOADED PDF LOADING
# ============================================================

def load_uploaded_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        try:
            pdf_bytes = uploaded_file.getvalue()

            documents.extend(
                read_pdf_bytes(
                    pdf_bytes,
                    uploaded_file.name,
                )
            )

        except Exception as error:
            st.warning(
                f"Could not read {uploaded_file.name}: {error}"
            )

    return documents


# ============================================================
# 9. SEMANTIC CONVERSATION MEMORY
# ============================================================

def retrieve_chat_memory(question, previous_messages):
    """
    Retrieve semantically relevant messages from this chat.

    This searches prior user questions and assistant answers.
    It does not search other conversations.
    """

    memories = []

    for index, message in enumerate(previous_messages):
        content = message.get("content", "").strip()

        if not content:
            continue

        if message.get("role") not in ("user", "assistant"):
            continue

        if content.startswith("Could not generate an answer:"):
            continue

        memories.append({
            "role": message["role"],
            "text": content,
            "index": index,
        })

    if not memories:
        return []

    memory_texts = tuple(
        item["text"] for item in memories
    )

    memory_vectors = cached_embeddings(memory_texts)
    question_vector = cached_embeddings((question,))[0]

    scores = np.dot(memory_vectors, question_vector)

    top_indices = np.argsort(scores)[::-1][:MEMORY_TOP_K]

    results = []

    for idx in top_indices:
        score = float(scores[idx])

        if score >= MEMORY_THRESHOLD:
            item = memories[int(idx)]

            results.append({
                "role": item["role"],
                "text": item["text"],
                "score": score,
                "index": item["index"],
            })

    results.sort(key=lambda item: item["index"])

    return results


# ============================================================
# 10. CONTEXTUAL QUESTION REWRITING
# ============================================================

def contextualize_question(question, previous_messages):
    """
    Rewrite a follow-up into a standalone retrieval query.
    """

    if not previous_messages:
        return question

    recent_history = previous_messages[-10:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
        if message.get("role") in ("user", "assistant")
    )

    client = get_groq_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite user questions for a PDF search "
                    "system. Use the conversation to resolve "
                    "references such as it, they, this, that, "
                    "the first one, and the previous topic. "
                    "Return a standalone question that preserves "
                    "the user's intended meaning. Do not answer "
                    "the question. Do not add unsupported details. "
                    "If the question is already standalone, return "
                    "it unchanged. Return only the rewritten query."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest question:\n{question}"
                ),
            },
        ],
        temperature=0,
        max_tokens=200,
    )

    rewritten = response.choices[0].message.content

    if rewritten and rewritten.strip():
        return rewritten.strip()

    return question


# ============================================================
# 11. PDF SEMANTIC RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    folder_chunks,
    folder_vectors,
    upload_chunks,
    upload_vectors,
):
    all_chunks = list(folder_chunks) + list(upload_chunks)

    if not all_chunks:
        return []

    vector_groups = []

    if len(folder_chunks):
        vector_groups.append(folder_vectors)

    if len(upload_chunks):
        vector_groups.append(upload_vectors)

    if not vector_groups:
        return []

    all_vectors = np.vstack(vector_groups)

    question_vector = cached_embeddings((question,))[0]

    # Embeddings are normalized, so dot product is cosine similarity.
    scores = np.dot(all_vectors, question_vector)

    best_indices = np.argsort(scores)[::-1][:TOP_K]

    return [
        {
            **all_chunks[int(index)],
            "score": float(scores[index]),
        }
        for index in best_indices
    ]


# ============================================================
# 12. GENERATE ANSWER USING PDF + MEMORY
# ============================================================

def generate_answer(
    question,
    results,
    previous_messages,
    memory_context=None,
):
    client = get_groq_client()

    if results:
        context = "\n\n".join(
            (
                f"Source: {item['source']}, "
                f"page {item['page']}\n"
                f"{item['text']}"
            )
            for item in results
        )
    else:
        context = "No relevant PDF passages were retrieved."

    if memory_context:
        memory_text = "\n\n".join(
            f"{item['role']}: {item['text']}"
            for item in memory_context
        )
    else:
        memory_text = "No semantically relevant memories retrieved."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful PDF RAG assistant with "
                "conversation memory. Answer the latest question "
                "using relevant PDF passages and conversation "
                "history. Resolve follow-up references correctly. "
                "Use conversation history to understand the user, "
                "but do not treat previous assistant answers as "
                "verified PDF evidence. Do not invent facts. "
                "If the answer is not supported by the PDFs, "
                "clearly say so. When possible, cite the PDF "
                "filename and page number. Be concise and helpful."
            ),
        }
    ]

    for message in previous_messages[-8:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    messages.append({
        "role": "user",
        "content": (
            f"RELEVANT PDF PASSAGES:\n{context}\n\n"
            f"SEMANTICALLY RETRIEVED CHAT MEMORIES:\n"
            f"{memory_text}\n\n"
            f"CURRENT USER QUESTION:\n{question}\n\n"
            "Answer the current question. Use the retrieved "
            "memories to understand references and prior topics. "
            "Prefer PDF passages for factual claims about the "
            "documents. If the PDF passages do not contain the "
            "answer, state this clearly rather than guessing."
        ),
    })

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    return (
        response.choices[0].message.content
        or "No answer was returned."
    )


# ============================================================
# 13. RESTORE CONVERSATIONS
# ============================================================

if "chats" not in st.session_state:
    try:
        st.session_state.chats = load_saved_chats()

    except Exception as error:
        st.error(f"Could not load saved chat history: {error}")
        st.info(
            "Check your Supabase secrets and the rag_chats table."
        )
        st.stop()


if (
    "active_chat" not in st.session_state
    or st.session_state.active_chat
    not in st.session_state.chats
):
    st.session_state.active_chat = None


if st.session_state.active_chat is None:
    if st.session_state.chats:
        st.session_state.active_chat = next(
            iter(st.session_state.chats)
        )
    else:
        try:
            create_new_chat()

        except Exception as error:
            st.error(f"Could not create a chat: {error}")
            st.stop()


# ============================================================
# 14. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("💬 Chat History")

    if st.button("＋ New chat", use_container_width=True):
        try:
            create_new_chat()
            st.rerun()

        except Exception as error:
            st.error(f"Could not create a new chat: {error}")

    st.divider()

    sorted_chat_items = sorted(
        st.session_state.chats.items(),
        key=lambda item: item[1].get("updated_at", ""),
        reverse=True,
    )

    for chat_id, saved_chat in sorted_chat_items:
        title = saved_chat.get("title") or "New chat"

        is_active = chat_id == st.session_state.active_chat
        label = ("● " if is_active else "") + title

        if st.button(
            label,
            key=f"history_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.active_chat = chat_id
            st.rerun()

    st.divider()

    st.caption("Answer model")
    st.write(LLM_MODEL)

    st.caption("Embedding model")
    st.write(EMBEDDING_MODEL)

    if st.button("Sign out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()


# ============================================================
# 15. MAIN APPLICATION
# ============================================================

st.title("📚 Personal PDF RAG Chatbot")

st.write(
    "Ask questions about your PDFs. Follow-up questions can "
    "use previous messages in the current conversation."
)

chat = st.session_state.chats[
    st.session_state.active_chat
]


# Load PDFs stored in the repository's data/ folder.
folder_signature = get_folder_signature()

with st.spinner("Loading project PDFs and extracting text..."):
    folder_chunks = load_folder_pdfs(
        str(DATA_DIR),
        folder_signature,
    )

    folder_texts = tuple(
        item["text"] for item in folder_chunks
    )

    folder_vectors = cached_embeddings(folder_texts)


with st.expander("📁 Project folder documents"):
    if folder_signature:
        for filename, _, _ in folder_signature:
            st.write(f"📄 {filename}")

        st.caption(
            f"Indexed text chunks: {len(folder_chunks)}"
        )
    else:
        st.info(
            "No PDFs found in data/. Add PDFs to GitHub's data "
            "folder or upload them below."
        )


# ============================================================
# 16. UPLOAD ADDITIONAL PDFS
# ============================================================

uploaded_files = st.file_uploader(
    "Upload additional PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"upload_{st.session_state.active_chat}",
)

upload_signature = tuple(
    (
        file.name,
        file.size,
        hashlib.sha256(file.getvalue()).hexdigest(),
    )
    for file in uploaded_files
)


if chat["upload_signature"] != upload_signature:
    with st.spinner(
        "Processing PDFs: extracting text and running OCR if needed..."
    ):
        chat["upload_signature"] = upload_signature

        chat["upload_chunks"] = load_uploaded_pdfs(
            uploaded_files
        )

        upload_texts = tuple(
            item["text"] for item in chat["upload_chunks"]
        )

        chat["upload_vectors"] = cached_embeddings(
            upload_texts
        )


if uploaded_files:
    st.caption(
        f"Uploaded PDF chunks: {len(chat['upload_chunks'])}"
    )


# ============================================================
# 17. DISPLAY CONVERSATION
# ============================================================

for message in chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")


# ============================================================
# 18. PROCESS CURRENT QUESTION
# ============================================================

question = st.chat_input("Ask a question about your PDFs...")


if question:
    # Snapshot history BEFORE appending the current question.
    previous_messages = list(chat["messages"])

    chat["messages"].append({
        "role": "user",
        "content": question,
    })

    if chat["title"] == "New chat":
        chat["title"] = question[:45] + (
            "..." if len(question) > 45 else ""
        )

    # Save the user's question.
    try:
        save_chat(st.session_state.active_chat, chat)
    except Exception as error:
        st.error(f"Could not save your question: {error}")

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching conversation memory and PDFs..."
            ):
                # A. Find semantically similar prior messages.
                memory_context = retrieve_chat_memory(
                    question,
                    previous_messages,
                )

                # B. Resolve references using recent history.
                search_question = contextualize_question(
                    question,
                    previous_messages,
                )

                # C. Search PDF chunks with the rewritten query.
                results = retrieve_chunks(
                    search_question,
                    folder_chunks,
                    folder_vectors,
                    chat["upload_chunks"],
                    chat["upload_vectors"],
                )

                # D. Generate answer using both sources.
                if not results and not memory_context:
                    answer = (
                        "I could not find relevant information "
                        "in the available PDFs or previous "
                        "conversation. Please rephrase your "
                        "question or upload a relevant PDF."
                    )
                    sources = []

                else:
                    answer = generate_answer(
                        question,
                        results,
                        previous_messages,
                        memory_context=memory_context,
                    )

                    sources = list(dict.fromkeys(
                        (
                            f"{item['source']} — "
                            f"page {item['page']}"
                        )
                        for item in results
                    ))

            st.markdown(answer)

            if sources:
                with st.expander("📖 Retrieved PDF sources"):
                    for source in sources:
                        st.write(f"- {source}")

            # Save the answer in this conversation.
            chat["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

            try:
                save_chat(st.session_state.active_chat, chat)

            except Exception as error:
                st.error(
                    f"Answer generated, but saving failed: {error}"
                )

        except Exception as error:
            error_message = (
                f"Could not generate an answer: {error}"
            )

            st.error(error_message)

            st.info(
                "Check Groq model availability, API limits, "
                "Supabase configuration, and Streamlit Secrets."
            )

            chat["messages"].append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
            })

            try:
                save_chat(st.session_state.active_chat, chat)

            except Exception as save_error:
                st.error(
                    f"Could not save the error message: {save_error}"
                )
