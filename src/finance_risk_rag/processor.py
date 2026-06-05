"""
Document processing and OCR module.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .exceptions import OCRError
from .llm import LLMClientWrapper
from .utils import get_file_hash, load_json_file, save_json_file


class DocumentProcessor:
    """Handles PDF text extraction, OCR, and classification."""

    def __init__(self, config=None) -> None:
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self.llm = LLMClientWrapper()

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

    def optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for better OCR results."""
        image = image.convert('L')
        image = image.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        return image

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """Extract text from PDF, using OCR as fallback."""
        text = f"# File: {pdf_path.name}\n\n"
        ocr_pages = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text += f"\n--- Page {i+1} ---\n{page_text}"
                    else:
                        img = page.to_image(resolution=self.config.ocr_dpi).original
                        img = self.optimize_image(img)
                        ocr_text = pytesseract.image_to_string(
                            img, lang=self.config.ocr_languages
                        )
                        text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                        ocr_pages += 1
        except Exception as e:
            self.logger.error(f"Failed to process {pdf_path}: {e}")
            raise OCRError(f"Failed to process {pdf_path}: {e}")

        return text, ocr_pages

    def classify_document(self, text_sample: str) -> Dict[str, Any]:
        """Classify document type using LLM."""
        if not self.llm.is_available:
            return {"type": "Unknown", "confidence": 0.0}

        prompt = f"""
Classify this financial document into one of these categories:
1. Audit Report, 2. Industry Report, 3. Company Research, 4. IPO Prospectus, 5. Financial Statement, 6. Other.

Text sample:
{text_sample[:2000]}

Respond ONLY in JSON: {{"type": "...", "confidence": 0.95, "reason": "..."}}
"""
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            # Basic JSON extraction from response
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except Exception:
            return {"type": "Unknown", "confidence": 0.0}

    def process_batch(self, docs_dir: Optional[Path] = None) -> None:
        """Batch process all PDFs in the documents directory."""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))
        log = load_json_file(self.config.processing_log_path)

        all_text = ""

        for pdf_path in pdf_files:
            file_hash = get_file_hash(pdf_path)
            cached = log.get(pdf_path.name, {})
            txt_path = pdf_path.with_suffix(".txt")

            if (cached.get("hash") == file_hash and
                cached.get("ocr_version") == self.config.ocr_version and
                txt_path.exists()):
                self.logger.info(f"Skipping {pdf_path.name} (already processed)")
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                self.logger.info(f"Processing {pdf_path.name}")
                text, ocr_count = self.extract_text_from_pdf(pdf_path)
                classification = self.classify_document(text[:3000])

                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                log[pdf_path.name] = {
                    "hash": file_hash,
                    "ocr_version": self.config.ocr_version,
                    "processed_at": datetime.now().isoformat(),
                    "classification": classification,
                    "ocr_pages": ocr_count
                }

            all_text += text + "\n\n" + "="*60 + "\n\n"

        # Save consolidated text
        with open(docs_dir / "all_extracted.txt", "w", encoding="utf-8") as f:
            f.write(all_text)

        save_json_file(log, self.config.processing_log_path)
