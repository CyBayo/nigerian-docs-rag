from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader
from rapidocr import RapidOCR

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

ocr_engine = RapidOCR()


def text_layer(pdf_path):
    reader = PdfReader(pdf_path)
    return [(page.extract_text() or "") for page in reader.pages]


def ocr_pages(pdf_path):
    pdf = pdfium.PdfDocument(str(pdf_path))
    pages = []
    for i in range(len(pdf)):
        image = pdf[i].render(scale=3).to_numpy()
        result = ocr_engine(image)
        lines = list(result.txts) if (result is not None and result.txts is not None) else []
        pages.append("\n".join(lines))
    return pages


for pdf_path in sorted(RAW.glob("*.pdf")):
    pages = text_layer(pdf_path)
    method = "text layer"
    if sum(len(p) for p in pages) < 100 * len(pages):
        pages = ocr_pages(pdf_path)
        method = "OCR"
    out_file = OUT / (pdf_path.stem + ".txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for number, text in enumerate(pages, start=1):
            f.write("=== Page " + str(number) + " ===\n" + text + "\n\n")
    total = sum(len(p) for p in pages)
    print(pdf_path.name + ": " + method + ", " + str(len(pages)) + " pages, " + str(total) + " characters")
