import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "GOOGLE_MODEL", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set.")
    
current_dir = Path(__file__).parent
pdf_path = current_dir / "design-patterns-pt-br.pdf"

docs = PyPDFLoader(str(pdf_path)).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    add_start_index=False,
).split_documents(docs)

if not text_splitter:
    raise SystemExit("No documents were created after text splitting.")

enriched = [
    Document(
        page_content=doc.page_content,
        metadata={
            k: v for k, v in doc.metadata.items() if v not in (None, "")
        },
    )
    for doc in text_splitter
]

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_MODEL", "gemini-embedding-001"))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True
)

store.add_documents(documents=enriched, ids=ids)