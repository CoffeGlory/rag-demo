import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

def chunk_text(text: str, chunk_size=800, overlap=120):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end-overlap)
    return chunks

pdf_path = r"E:\projects\rag-demo\data\NASM_CPT7_Study_Guide.pdf"
db_path = r"E:\projects\rag-demo\chroma_db"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path=db_path)
collection = client.get_or_create_collection(name="docs")

reader = PdfReader(pdf_path)

all_docs = []
all_metas = [] #metasdata for source tracking
all_ids = [] #ids for deletion if re-running

base = os.path.basename(pdf_path)
for page_idx, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if not text.strip():
        continue   
    chunks = chunk_text(text)
    for chunk_idx, chunk in enumerate(chunks):
        doc_id = f"{base}_page{page_idx+1}-c{chunk_idx}"
        all_ids.append(doc_id)
        all_docs.append(chunk)
        all_metas.append({
            "source": base,
            "page": page_idx + 1,
            "chunk": chunk_idx
        })

# get embeddings
embeddings = embed_model.encode(all_docs).tolist()
try:
    collection.delete(ids=all_ids)
except Exception:
    pass   

collection.add(
    ids = all_ids,
    documents=all_docs,
    metadatas=all_metas,
    embeddings=embeddings,
)

print("PDF:", base)
print("Pages processed:", len(reader.pages))
print("Chunks stored:", len(all_docs))