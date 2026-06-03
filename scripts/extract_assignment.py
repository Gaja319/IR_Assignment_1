from docx import Document
import sys
from pathlib import Path


def extract_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_assignment.py <docx-path> [out.txt]")
        sys.exit(1)
    docx_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "assignment_extracted.txt"
    text = extract_docx(docx_path)
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"Extracted text written to {out_path}")


if __name__ == '__main__':
    main()
