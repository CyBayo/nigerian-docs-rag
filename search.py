import sys

import chromadb
from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
query = " ".join(sys.argv[1:]) or "At what age can someone enrol for a BVN?"

embedder = TextEmbedding(model_name=MODEL)
vector = list(embedder.query_embed([query]))[0].tolist()

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("cbn_circulars")
results = collection.query(query_embeddings=[vector], n_results=3)

print(f"Question: {query}\n")
for doc, meta, dist in zip(
    results["documents"][0], results["metadatas"][0], results["distances"][0]
):
    print(f"[{meta['doc_id']} p{meta['page']}] distance={dist:.3f}")
    print(doc[:300])
    print()