"""
Finance-Risk-RAG Processor Module
=================================

Handles document extraction, OCR, and classification.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .engine import LLMClientWrapper
from .utils import get_file_hash, load_json, save_json, setup_logger


class DocumentProcessor:
    """Processes PDF documents using OCR and AI classification."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("DocumentProcessor")
        self.llm = LLMClientWrapper(self.config)

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

    def optimize_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results."""
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF, falling back to OCR if needed."""
        text = f"# Source: {pdf_path.name}\n\n"

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    text += f"\n--- Page {i+1} ---\n{page_text}"
                else:
                    # OCR fallback
                    img = page.to_image(resolution=self.config.ocr_dpi).original
                    img = self.optimize_image(img)
                    ocr_text = pytesseract.image_to_string(img, lang=self.config.ocr_languages)
                    text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"

        return text

    def classify_document(self, text: str) -> Dict[str, Any]:
        """Classify document type using LLM."""
        prompt = (
            "Classify this document into one of: Audit, Industry, Company Research, "
            "Listing Manual, Financial Report, Other. Return JSON with 'type' and "
            f"'confidence'.\n\nContent sample: {text[:2000]}"
        )

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.call(messages)

        # Simple extraction of JSON from response
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        return {"type": "Unknown", "confidence": 0.0}

    def process_batch(self, docs_dir: Optional[Path] = None):
        """Process all PDFs in a directory."""
        docs_dir = docs_dir or self.config.docs_dir
        log_path = self.config.cache_dir / "processing_log.json"
        log = load_json(log_path)

        results = {}
        for pdf_file in docs_dir.glob("*.pdf"):
            file_hash = get_file_hash(pdf_file)
            cached = log.get(pdf_file.name, {})

            if cached.get("hash") == file_hash and cached.get("version") == self.config.ocr_version:
                self.logger.info(f"Skipping {pdf_file.name} (cached)")
                continue

            self.logger.info(f"Processing {pdf_file.name}...")
            text = self.extract_text(pdf_file)
            classification = self.classify_document(text)

            # Save extracted text
            txt_path = pdf_file.with_suffix(".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            log[pdf_file.name] = {
                "hash": file_hash,
                "version": self.config.ocr_version,
                "classification": classification,
            }
            results[pdf_file.name] = classification

        save_json(log, log_path)
        return results
