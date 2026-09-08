import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "my_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.6-flash"

# Lazy-loaded singleton for embedding model
_embeddings_singleton = None


def get_embeddings():
    """
    Lazy singleton loader for HuggingFace embedding model.
    Runs locally on CPU and generates 384-dimensional normalized vectors.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def ingest_pdf(
    file_path,
    reset=True,
    persist_directory=CHROMA_DB_PATH,
    collection_name=COLLECTION_NAME,
):
    """
    Parses a PDF file, splits it into chunks, embeds them, and persists into ChromaDB.

    Parameters:
    - file_path: Path to the target PDF file on disk.
    - reset: If True, clears existing collection first. If False, appends to it.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    docs = PyPDFLoader(file_path).load()
    if not docs:
        raise ValueError(
            "No extractable text found in this PDF. It may be empty or scanned."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()

    if reset and os.path.exists(persist_directory):
        try:
            Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=embeddings,
            ).delete_collection()
        except Exception:
            pass

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return len(chunks), vectorstore


def get_vectorstore(persist_directory=CHROMA_DB_PATH, collection_name=COLLECTION_NAME):
    """Returns the Chroma vectorstore connected to the persistent database."""
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )


def get_retriever(k=4, fetch_k=10, lambda_mult=0.5):
    """Returns an MMR retriever over the Chroma vectorstore."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def get_rag_prompt():
    """Returns the strict grounded RAG prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )


def content_to_text(content):
    """Extracts displayable text from plain strings or Gemini content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(content_to_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return "" if content is None else str(content)


def get_llm(api_key=None):
    """Returns a Google Gemini chat model configured for this project."""
    key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=key,
    )


def interactive_cli():
    """Starts the interactive Terminal CLI Q&A interface."""
    retriever = get_retriever()
    llm = get_llm()
    prompt = get_rag_prompt()

    print("====================================")
    print("     RAG BOOK ASSISTANT CLI")
    print("====================================")
    print(f"Embedding Model : {EMBEDDING_MODEL}")
    print(f"Vector Database : ChromaDB ({CHROMA_DB_PATH})")
    print(f"LLM Model       : {LLM_MODEL}")
    print("====================================")
    print("Type your question below (or '0' / 'exit' to quit):")

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query in ("0", "exit", "quit"):
            print("Exiting...")
            break

        if not query:
            continue

        docs = retriever.invoke(query)
        if not docs:
            print("\nAI: I could not find the answer in the document.")
            continue

        context = "\n\n".join([doc.page_content for doc in docs])
        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)

        print(f"\nAI: {content_to_text(response.content)}")


def main():
    """Entry point for CLI commands (ingestion or chat)."""
    parser = argparse.ArgumentParser(
        description="RAG Book Assistant — Ingest documents or query knowledge base."
    )
    parser.add_argument(
        "--ingest",
        type=str,
        metavar="PDF_PATH",
        help="Path to a PDF file to ingest into the vector database",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append documents to the existing knowledge base without resetting",
    )

    args = parser.parse_args()

    if args.ingest:
        print(f"Reading and ingesting '{args.ingest}' ...")
        try:
            count, _ = ingest_pdf(args.ingest, reset=not args.append)
            action = "Appended to" if args.append else "Rebuilt"
            print(f"Success: {action} knowledge base with {count} chunks.")
            print("You can now run: python main.py or streamlit run app.py")
        except Exception as exc:
            print(f"Error during ingestion: {exc}")
            sys.exit(1)
    else:
        interactive_cli()


if __name__ == "__main__":
    main()
