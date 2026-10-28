from pathlib import Path
from pypdf import PdfReader

for pdf in sorted(Path("data/raw").glob("*.pdf")):
    reader = PdfReader(pdf)
    text = "".join((page.extract_text() or "") for page in reader.pages)
    print(f"{pdf.name}: {len(reader.pages)} pages, {len(text)} characters")
