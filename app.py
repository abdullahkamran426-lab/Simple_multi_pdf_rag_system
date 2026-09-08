import os
import tempfile
import importlib

import streamlit as st

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    Chroma = importlib.import_module("langchain_chroma").Chroma
except ImportError as exc:
    raise ImportError(
        "The 'langchain-chroma' package is required. Install it with "
        "'pip install -U langchain-chroma'."
    ) from exc

ChatPromptTemplate = importlib.import_module(
    "langchain_core.prompts"
).ChatPromptTemplate

try:
    ChatGoogleGenerativeAI = importlib.import_module(
        "langchain_google_genai"
    ).ChatGoogleGenerativeAI
except ImportError as exc:
    raise ImportError(
        "The 'langchain-google-genai' package is required. Install it with "
        "'pip install -U langchain-google-genai'."
    ) from exc

from main import (
    ingest_pdf,
    get_embeddings as _get_embeddings,
    content_to_text,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")
LLM_MODEL = "gemini-3.6-flash"


# ============================================================
# CACHED RESOURCES
# (loaded once per server process instead of on every rerun)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Reuses the exact same embedding model/config as ingest.py and main.py.
    return _get_embeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


def stream_text(stream):
    """Converts LangChain message chunks into text for Streamlit's typewriter."""
    for chunk in stream:
        text = content_to_text(getattr(chunk, "content", chunk))
        if text:
            yield text


def load_existing_db():
    """
    If a persisted knowledge base already exists on disk (e.g. built via
    `python ingest.py book.pdf` or a previous app run), attach it —
    so the person doesn't have to re-upload on every restart.
    """
    if not os.path.exists(CHROMA_DB_PATH):
        return None

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=get_embeddings(),
    )
    try:
        count = len(vs.get()["ids"])
    except Exception:
        count = 0

    return (vs, count) if count > 0 else None


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "vectorstore": None,
    "book_name": None,
    "chunk_count": 0,
    "chat_history": [],
    "checked_existing_db": False,
    "indexed_docs": [],          # names of every PDF ingested this session
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Auto-attach a pre-existing shared knowledge base, once per session.
if st.session_state.vectorstore is None and not st.session_state.checked_existing_db:
    st.session_state.checked_existing_db = True
    existing = load_existing_db()
    if existing:
        vs, count = existing
        st.session_state.vectorstore = vs
        st.session_state.book_name = "Existing knowledge base"
        st.session_state.chunk_count = count
        if not st.session_state.indexed_docs:
            st.session_state.indexed_docs = ["Existing knowledge base"]


def reset_book():
    """Detach the current knowledge base from this browser session.

    Note: this does NOT delete the files on disk — main.py and other
    sessions can still use them. The on-disk collection is only replaced
    when a new PDF is ingested (see ingest_pdf's reset=True default).
    """
    st.session_state.vectorstore = None
    st.session_state.book_name = None
    st.session_state.chunk_count = 0
    st.session_state.chat_history = []
    st.session_state.indexed_docs = []


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful reading assistant. Use ONLY the provided "
            "context to answer the question. Do not use outside knowledge. "
            "If the answer is not present in the context, respond with "
            'exactly: "I could not find the answer in the document."',
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}",
        ),
    ]
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --bg: #101019;
        --card-bg: #181822;
        --bg-parchment: #1e1820;
        --ink: #edeef3;
        --muted: #9799ac;
        --accent: #d8697c;
        --accent-soft: #2a1c22;
        --accent-line: #e98ba0;
        --border: #2b2c3a;
        --success: #34d399;
        --success-soft: #10291f;
        --radius-lg: 18px;
        --radius-md: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, .main-title, .card-title, .answer-title {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.01em;
    }

    .stApp {
        background: var(--bg);
    }

    /* ---------- Header ---------- */

    .main-header {
        padding: 18px 0 22px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 28px;
    }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 500;
        margin-bottom: 6px;
    }

    .main-title {
        font-size: 40px;
        font-weight: 600;
        margin: 0 0 8px 0;
        color: var(--ink);
    }

    .subtitle {
        font-size: 15.5px;
        color: var(--muted);
        max-width: 640px;
        line-height: 1.6;
    }

    /* ---------- Spine cards (signature element) ---------- */

    .card {
        background: var(--card-bg);
        padding: 22px 24px;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        box-shadow: 0 2px 10px rgba(30, 36, 56, 0.04);
        margin-bottom: 18px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 600;
        margin-bottom: 6px;
        color: var(--ink);
    }

    .card-text {
        color: var(--muted);
        font-size: 14.5px;
        line-height: 1.55;
    }

    /* ---------- Status pill ---------- */

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: var(--success-soft);
        color: var(--success);
        font-weight: 600;
        font-size: 13.5px;
        margin-bottom: 14px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--muted);
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: var(--radius-md);
        font-weight: 600;
        padding: 10px 18px;
        border: 1px solid var(--border);
    }

    .stButton > button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-line);
        border-color: var(--accent-line);
    }

    /* ---------- Inputs ---------- */

    .stTextInput input, [data-testid="stChatInput"] textarea {
        border-radius: var(--radius-md);
        background: var(--card-bg);
        color: var(--ink);
        border: 1px solid var(--border);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-parchment);
        border: 1px dashed var(--border);
        border-radius: var(--radius-md);
    }

    /* ---------- Chat ---------- */

    [data-testid="stChatMessage"] {
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        padding: 4px 6px;
    }

    /* ---------- Metrics ---------- */

    [data-testid="stMetric"] {
        background: var(--bg-parchment);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 12px 14px;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 12.5px;
        margin-top: 46px;
        padding: 18px;
        font-family: 'JetBrains Mono', monospace;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 20px 0;">
            <div style="font-size:46px;">📚</div>
            <h2 style="margin:6px 0 2px 0; font-size:22px;">RAG Book Assistant</h2>
            <p style="color:var(--muted); font-size:13.5px; margin:0;">
                AI-powered document Q&amp;A
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if not GOOGLE_API_KEY:
        st.warning(
            "⚠️ **GOOGLE_API_KEY** not found in your environment.\n\n"
            "Add it to a `.env` file to enable answer generation.",
            icon="⚠️",
        )

    st.markdown("### ⚙️ System")
    st.info(
        f"""
        **Embedding:** Hugging Face
        **Vector DB:** ChromaDB (shared, on disk)
        **LLM:** {LLM_MODEL}
        **Retrieval:** MMR · **Answers:** Streamed
        """
    )

    st.divider()

    st.markdown("### 📖 How it works")
    st.markdown(
        """
        1. Upload one or more PDFs
        2. Build the knowledge base
        3. Ask questions in the chat
        4. Relevant passages are retrieved
        5. The model streams an answer
        """
    )

    # ── Active knowledge base panel ──────────────────────────────────────
    if st.session_state.vectorstore is not None:
        st.divider()
        st.markdown("### 📚 Knowledge Base")
        st.caption(f"🧩 **{st.session_state.chunk_count}** chunks indexed")

        for i, doc_name in enumerate(st.session_state.indexed_docs, 1):
            st.caption(f"{i}. 📄 {doc_name}")

        st.write("")

        # ── Add another PDF without losing existing data ──────────────
        with st.expander("➕ Add another PDF"):
            extra_file = st.file_uploader(
                "Append to knowledge base",
                type=["pdf"],
                key="sidebar_uploader",
                label_visibility="collapsed",
            )
            if extra_file:
                if st.button("📥 Append to Knowledge Base", use_container_width=True):
                    _file_path = None
                    with st.spinner(f"Adding {extra_file.name}…"):
                        try:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".pdf"
                            ) as _tmp:
                                _tmp.write(extra_file.getvalue())
                                _file_path = _tmp.name

                            new_chunks, new_vs = ingest_pdf(_file_path, reset=False)
                            st.session_state.vectorstore = new_vs
                            st.session_state.chunk_count += new_chunks
                            st.session_state.indexed_docs.append(extra_file.name)
                            st.success(
                                f"✓ Added **{extra_file.name}** — "
                                f"{new_chunks} new chunks indexed."
                            )
                            st.rerun()
                        except Exception as _exc:
                            st.error(f"Failed to add PDF: {_exc}")
                        finally:
                            if _file_path and os.path.exists(_file_path):
                                os.unlink(_file_path)

        st.write("")
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            reset_book()
            st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="eyebrow">Retrieval-Augmented Generation</div>
        <div class="main-title">📚 RAG Book Assistant</div>
        <div class="subtitle">
            Upload one or more books, build a searchable knowledge base, and ask
            questions grounded directly in your documents — no hallucinated
            answers, only what the text actually says.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOAD SECTION
# ============================================================

if st.session_state.vectorstore is None:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">📄 Upload Your Documents</div>
            <div class="card-text">
                Upload one or more PDF documents to create your AI-powered knowledge base.
                You can add more documents at any time from the sidebar.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        total_size = sum(f.size for f in uploaded_files)

        st.markdown(
            f"""
            <div class="status-pill">✓ {len(uploaded_files)} PDF(s) ready to index</div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Files", len(uploaded_files))
        with col2:
            st.metric("📦 Total Size", f"{total_size / 1024:.1f} KB")
        with col3:
            st.metric("📚 Type", "PDF Document")

        st.write("")

        if st.button(
            "🚀 Build Knowledge Base",
            use_container_width=True,
            type="primary",
        ):
            progress = st.progress(0)
            status_text = st.empty()
            total_chunks = 0
            final_vs = None
            ingested_names = []

            for idx, uploaded_file in enumerate(uploaded_files):
                file_path = None
                status_text.info(
                    f"📖 Processing **{uploaded_file.name}** "
                    f"({idx + 1} / {len(uploaded_files)})…"
                )
                progress.progress(int((idx / len(uploaded_files)) * 90) + 5)

                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_path = tmp_file.name

                    # First file resets the DB; subsequent ones append.
                    chunk_count, final_vs = ingest_pdf(
                        file_path, reset=(idx == 0)
                    )
                    total_chunks += chunk_count
                    ingested_names.append(uploaded_file.name)

                except Exception as exc:
                    st.warning(f"⚠️ Skipped **{uploaded_file.name}**: {exc}")

                finally:
                    if file_path and os.path.exists(file_path):
                        os.unlink(file_path)

            if final_vs is not None:
                progress.progress(100)
                st.session_state.vectorstore = final_vs
                st.session_state.book_name = (
                    ingested_names[0]
                    if len(ingested_names) == 1
                    else f"{len(ingested_names)} documents"
                )
                st.session_state.chunk_count = total_chunks
                st.session_state.chat_history = []
                st.session_state.indexed_docs = ingested_names

                status_text.empty()
                progress.empty()
                st.success(
                    f"✓ Knowledge base ready — {total_chunks} chunks from "
                    f"{len(ingested_names)} document(s). "
                    f"Also available to `main.py` from now on."
                )
                st.balloons()
                st.rerun()
            else:
                status_text.empty()
                progress.empty()
                st.error("No documents could be processed successfully.")


# ============================================================
# CHAT / RAG SECTION
# ============================================================

if st.session_state.vectorstore is not None:

    doc_count = len(st.session_state.indexed_docs)
    doc_label = (
        st.session_state.indexed_docs[0]
        if doc_count == 1
        else f"{doc_count} documents"
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">💬 Ask your knowledge base</div>
            <div class="card-text">
                <strong>{st.session_state.chunk_count}</strong> chunks indexed
                across <strong>{doc_label}</strong>.
                Ask anything about the content of your uploaded documents.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    # Render existing conversation turns
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"], avatar=turn.get("avatar")):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(f"📚 View sources ({len(turn['sources'])})"):
                    for i, src in enumerate(turn["sources"]):
                        page = src.get("page")
                        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                        st.markdown(f"**Source {i + 1}** · {page_label}")
                        st.caption(src["text"])
                        if i < len(turn["sources"]) - 1:
                            st.divider()

    query = st.chat_input("e.g. What is the main idea of this book?")

    if query:
        st.session_state.chat_history.append(
            {"role": "user", "content": query, "avatar": "🧑‍💻"}
        )
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="📚"):
            if not GOOGLE_API_KEY:
                answer = (
                    "I can't generate an answer because **GOOGLE_API_KEY** "
                    "isn't set. Add it to your `.env` file and restart the app."
                )
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "avatar": "📚"}
                )
            else:
                with st.spinner("🔎 Searching the knowledge base…"):
                    try:
                        retrieved = retriever.invoke(query)
                    except Exception as exc:
                        retrieved = []
                        st.error(f"Retrieval failed: {exc}")

                if not retrieved:
                    answer = "I could not find the answer in the document."
                    st.markdown(answer)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "avatar": "📚"}
                    )
                else:
                    context = "\n\n".join(doc.page_content for doc in retrieved)
                    final_prompt = ANSWER_PROMPT.invoke(
                        {"context": context, "question": query}
                    )

                    # ── Streamed response ─────────────────────────────────────
                    # st.write_stream() renders each token as it arrives,
                    # giving the user immediate visual feedback without a spinner.
                    try:
                        llm = get_llm()
                        answer = st.write_stream(stream_text(llm.stream(final_prompt)))
                    except Exception as exc:
                        answer = (
                            "I ran into an error while generating the "
                            f"answer: {exc}"
                        )
                        st.markdown(answer)

                    sources = [
                        {
                            "text": doc.page_content[:1000],
                            "page": doc.metadata.get("page"),
                        }
                        for doc in retrieved
                    ]

                    with st.expander(f"📚 View sources ({len(sources)})"):
                        for i, src in enumerate(sources):
                            page = src.get("page")
                            page_label = (
                                f"Page {page + 1}"
                                if isinstance(page, int)
                                else "Page unknown"
                            )
                            st.markdown(f"**Source {i + 1}** · {page_label}")
                            st.caption(src["text"])
                            if i < len(sources) - 1:
                                st.divider()

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "avatar": "📚",
                            "sources": sources,
                        }
                    )


# ============================================================
# EMPTY STATE
# ============================================================

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding:50px 30px; border-left-width:4px;">
            <div style="font-size:56px;">📖</div>
            <h2 style="margin:10px 0 6px 0;">Your AI Book Assistant</h2>
            <p style="color:var(--muted); font-size:15px; max-width:420px; margin:0 auto;">
                Upload one or more PDFs above to build your personal AI-powered
                knowledge base.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RAG Book Assistant · Hugging Face · ChromaDB · Mistral AI
    </div>
    """,
    unsafe_allow_html=True,
)
# End of Streamlit Application
