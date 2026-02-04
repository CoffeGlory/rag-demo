from sentence_transformers import SentenceTransformer
import chromadb

#1 local embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

#2 local ChromaDB
db_path = r"E:\projects\rag-demo\chroma_db"
client = chromadb.PersistentClient(path=db_path)

collection = client.get_or_create_collection(name="docs")

docs = [
    "RAG stands for Retrieval-Augmented Generation.",
    "RAG combines vector search with language models.",
    "Embeddings convert text into numerical vectors.",
    "Vector databases enable semantic search."
]

# 4) initialize collection (clear existing data)
ids = [f"doc{i}" for i in range(len(docs))]
metadatas = [{"source": "demo", "chunk": i} for i in range(len(docs))] # optional metadata for source tracking
#get embeddings
embeddings = model.encode(docs).tolist()

try:
    collection.delete(ids=ids)
except Exception:
    pass

collection.add(
    documents=docs, 
    embeddings=embeddings,
    metadatas=metadatas, 
    ids=ids)

print("Stored", len(docs), "documents")
print("DB path:", db_path)
print("Collection name:", collection.name)
