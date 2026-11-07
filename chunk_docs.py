import json
import re
from pathlib import Path

PROCESSED = Path("data/processed")
OUT = Path("data/chunks.jsonl")
MAX_CHARS = 800
OVERLAP = 150


def split_pages(text):
    parts = re.split(r"=== Page (\d+) ===\n", text)
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + MAX_CHARS // 2, end)
            if cut != -1:
                end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
        space = text.find(" ", start)
        if space != -1 and space < end:
            start = space + 1
    return chunks


records = []
for path in sorted(PROCESSED.glob("*.txt")):
    doc_id = path.stem
    text = path.read_text(encoding="utf-8")
    count = 0
    for page_number, page_text in split_pages(text):
        for chunk in chunk_text(clean(page_text)):
            count += 1
            records.append(
                {
                    "chunk_id": f"{doc_id}::p{page_number}::c{count}",
                    "doc_id": doc_id,
                    "page": page_number,
                    "text": chunk,
                }
            )
    print(f"{doc_id}: {count} chunks")

with open(OUT, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Total: {len(records)} chunks saved to {OUT}")
if records:
    print(json.dumps(records[min(2, len(records) - 1)], indent=2, ensure_ascii=False))