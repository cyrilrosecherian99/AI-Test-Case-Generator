import os
import shutil
import hashlib

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_PERSIST_DIR = "./vectordb"


# ---------- Utility Functions ----------
def get_md5(text: str):
    """Create a unique hash for each document chunk."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_all_files_from_folder(folder_path: str):
    """Recursively get all PDF and Word files from a folder."""
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".pdf", ".docx", ".doc")):
                all_files.append(os.path.join(root, file))
    print(f"Found {len(all_files)} files in '{folder_path}'")
    return all_files


def prepare_docs(file_paths):
    """Load and split PDFs or Word documents into chunks."""
    # Normalize input: accept a folder path, a single file path, or a list of file paths
    if isinstance(file_paths, str):
        if os.path.isdir(file_paths):
            file_paths = get_all_files_from_folder(file_paths)
        else:
            file_paths = [file_paths]

    all_docs = []
    for file in file_paths:
        print(f"Loading: {file}")
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            loader = UnstructuredWordDocumentLoader(file)

        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(file)
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} raw documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    for doc in split_docs:
        doc.metadata["doc_id"] = get_md5(doc.page_content)

    print(f"Split into {len(split_docs)} chunks")
    return split_docs


# ---------- Chroma Ingestion ----------
def add_to_chroma(docs, collection_name, persist_dir):
    """Store document chunks in a Chroma vector DB using Azure OpenAI embeddings."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large", #os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),  # your embedding deployment name
        azure_endpoint="https://eyq-incubator.america.fabric.ey.com/eyq/us/api", #os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OPENAI_KEY"),
        api_version="2025-04-01-preview",
        model="text-embedding-3-large"
    )

    # --- Create or reuse a persistent Chroma client to avoid inconsistent settings errors ---
    CHROMA_TENANT = os.getenv("CHROMA_TENANT") or "default_tenant"
    CHROMA_DATABASE = os.getenv("CHROMA_DATABASE") or "default_database"

    # Try to create a PersistentClient with tenant/database. If a client for this path
    # already exists with other settings, fall back to a simpler client creation to
    # allow operations to continue in the same process.
    try:
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    except Exception as e:
        # Handle the specific case where a Chroma instance already exists for this path
        err_msg = str(e)
        print(f"Warning creating PersistentClient with tenant/database: {err_msg}")
        print("Falling back to PersistentClient without tenant/database (to reuse existing instance if possible).")
        try:
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        except Exception as e2:
            # If fallback also fails, surface the original error for troubleshooting
            raise RuntimeError(f"Failed to create or reuse a Chroma client for '{persist_dir}': {e2}")

    # Ensure collection exists and use the explicit client in LangChain's Chroma wrapper
    try:
        client.get_or_create_collection(name=collection_name)
    except Exception:
        pass

    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    MAX_BATCH = 5000
    for i in range(0, len(docs), MAX_BATCH):
        batch = docs[i:i + MAX_BATCH]
        print(f"Inserting batch {i // MAX_BATCH + 1} ({len(batch)} docs)...")
        vectordb.add_documents(batch)

    print(f"✅ Successfully ingested {len(docs)} docs into '{collection_name}' at {persist_dir}")


def _safe_remove_dir(path):
    """Try to remove a directory. If files are locked on Windows, raise a helpful error."""
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except Exception as e:
        # Provide a clearer message for locked files on Windows (WinError 32)
        raise RuntimeError(
            f"Failed to remove existing persist directory '{path}'. It may be in use by another process (e.g. a running app). "
            "Please close any processes that may be using the DB (look for 'chroma.sqlite3') and try again. Original error: "
            + str(e)
        )


# ---------- Ingestion Functions ----------
def ingest_domain_knowledge_to_chroma(domain_folder):
    """Store domain-related documents."""
    persist_dir = os.path.join(BASE_PERSIST_DIR, "domain_knowledge")
    collection_name = "domain_contexts"

    if os.path.exists(persist_dir):
        print(f"Removing old domain knowledge DB at {persist_dir}")
        _safe_remove_dir(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    domain_files = get_all_files_from_folder(domain_folder)
    docs = prepare_docs(domain_files)
    add_to_chroma(docs, collection_name, persist_dir)

def ingest_requirement_docs_to_chroma(requirement_files):
    """Store requirement-specific documents (safe, no DB deletion)."""

    persist_dir = os.path.join(BASE_PERSIST_DIR, "requirement_docs")
    collection_name = "requirement_contexts"

    # ✅ Do NOT delete existing DB
    os.makedirs(persist_dir, exist_ok=True)

    # Prepare docs
    try:
        docs = prepare_docs(requirement_files)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Document loading failed: {e}")

    # ✅ Add to existing DB (no lock issue)
    add_to_chroma(docs, collection_name, persist_dir)

    print(f"✅ Ingestion completed successfully into {persist_dir}")





# ---------- Main ----------
if __name__ == "__main__":
    domain_folder = r"context_files/Guidewire_Policy_Center_TrainingDocs"
    requirement_files = [r"context_files/Credit_Card_Login_Requirements.pdf"]

    ingest_domain_knowledge_to_chroma(domain_folder)
    ingest_requirement_docs_to_chroma(requirement_files)
