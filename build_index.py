import json
from pathlib import Path

import chromadb
from fastembed import TextEmbedding

CHUNKS = Path("data/chunks.jsonl")
MODEL = "BAAI/bge-small-en-v1.5"

records = [
    json.loads(line)
    for line in CHUNKS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

embedder = TextEmbedding(model_name=MODEL)
texts = [r["text"] for r in records]
vectors = [v.tolist() for v in embedder.embed(texts)]

client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("cbn_circulars")
except Exception:
    pass
collection = client.create_collection(name="cbn_circulars")

collection.add(
    ids=[r["chunk_id"] for r in records],
    documents=texts,
    embeddings=vectors,
    metadatas=[{"doc_id": r["doc_id"], "page": r["page"]} for r in records],
)

print(f"Indexed {collection.count()} chunks")