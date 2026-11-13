import os
import sys

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not found. Check that .env exists in this folder.")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "openai/gpt-oss-20b"
TOP_K = 4
MAX_DISTANCE = 0.7  # starting guess; we will tune it with the evaluation set

SYSTEM_PROMPT = (
    "You answer questions about Central Bank of Nigeria circulars using ONLY the passages provided. "
    "Rules: "
    "1) Use only facts stated in the passages. If they do not contain the answer, say you cannot find it in the provided circulars. "
    "2) Cite every claim with its source tag in square brackets, exactly as given, for example [DOC-ID_2026-05-29 p1]. "
    "3) Each source tag ends with the document date. If two passages give different values for the same thing, "
    "use the one from the later-dated document and say that it replaced or amended the earlier one. "
    "4) Keep answers short and plain. "
    "5) You are not a lawyer. Do not give legal advice."
)

question = " ".join(sys.argv[1:]).strip()
if not question:
    sys.exit('Usage: python answer.py "your question"')

embedder = TextEmbedding(model_name=EMBED_MODEL)
vector = list(embedder.query_embed([question]))[0].tolist()
collection = chromadb.PersistentClient(path="chroma_db").get_collection("cbn_circulars")
results = collection.query(query_embeddings=[vector], n_results=TOP_K)

kept = [
    (doc, meta, dist)
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )
    if dist <= MAX_DISTANCE
]

print(f"Question: {question}\n")

if not kept:
    print("I can't find this in the CBN circulars I have, so I won't guess.")
else:
    context = "\n\n".join(f"[{m['doc_id']} p{m['page']}]\n{d}" for d, m, _ in kept)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    print((response.choices[0].message.content or "(empty reply)").strip())
    print("\nPassages used:")
    for _, meta, dist in kept:
        print(f"  - {meta['doc_id']} p{meta['page']} (distance {dist:.3f})")