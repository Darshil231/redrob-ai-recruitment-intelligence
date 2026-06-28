"""Job-description parser for recruiter-uploaded text and DOCX files."""

from pathlib import Path


class JDParser:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> str:
        suffix = self.filepath.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self.filepath.read_text(encoding="utf-8")

        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Install python-docx to parse .docx job descriptions.") from exc

        doc = Document(str(self.filepath))

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
