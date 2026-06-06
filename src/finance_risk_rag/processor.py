"""
Finance-Risk-RAG 文档处理模块
============================
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .exceptions import OCRError
from .llm import LLMClientWrapper
from .models import ClassificationResult
from .utils import get_file_hash, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """处理 PDF 文档并提取文本"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm_client = LLMClientWrapper()
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def classify_document(self, text_sample: str) -> ClassificationResult:
        if not self.llm_client.is_available:
            return ClassificationResult(type="未知", confidence=0.0, reason="LLM unavailable")

        prompt = f"""
请判断以下财务文档属于哪一类？输出 JSON 格式。

文本样本：
{text_sample[:3000]}

【可选类别】
1. 审计报告
2. 行业报告
3. 公司研究报告
4. 上市手册
5. 财报
6. 其他

【输出要求】
- 仅输出 JSON 格式
- 示例：{{"type": "行业报告", "confidence": 0.93, "reason": "..."}}
"""
        try:
            import json

            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.2)
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return ClassificationResult(
                type=data.get("type", "其他"),
                confidence=data.get("confidence", 0.0),
                reason=data.get("reason", ""),
            )
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(type="未知", confidence=0.0, reason=str(e))

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        text += f"\n--- Page {i+1} (Text) ---\n{page_text}"
                    else:
                        img = page.to_image(resolution=self.config.ocr_dpi).original
                        img = self.optimize_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(
                            img, lang=self.config.ocr_languages, config="--oem 1 --psm 3"
                        )
                        text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                        ocr_pages += 1
            return text, ocr_pages
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise OCRError(f"PDF extraction failed for {pdf_path}: {e}")

    def process_directory(self, docs_dir: Optional[Path] = None):
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))
        log = load_json_file(self.config.processing_log_path)

        all_text = ""
        classifications = {}

        for pdf_path in pdf_files:
            file_hash = get_file_hash(pdf_path)
            cached = log.get(pdf_path.name, {})
            txt_path = pdf_path.with_suffix(".txt")

            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self.config.ocr_version
                and txt_path.exists()
            ):
                logger.info(f"Skipping {pdf_path.name} (cached)")
                text = txt_path.read_text(encoding="utf-8")
                classification_data = cached.get("classification", {"type": "未知"})
                classification = ClassificationResult(
                    type=classification_data.get("type", "未知"),
                    confidence=classification_data.get("confidence", 0.0),
                    reason=classification_data.get("reason", ""),
                )
            else:
                logger.info(f"Processing {pdf_path.name}")
                text, ocr_count = self.extract_text_from_pdf(pdf_path)
                txt_path.write_text(text, encoding="utf-8")
                classification = self.classify_document(text)

                log[pdf_path.name] = {
                    "hash": file_hash,
                    "ocr_version": self.config.ocr_version,
                    "classification": classification.to_dict(),
                    "ocr_pages": ocr_count,
                }

            all_text += text + "\n\n" + "=" * 60 + "\n\n"
            classifications[pdf_path.name] = classification.to_dict()

        (docs_dir / "all_extracted.txt").write_text(all_text, encoding="utf-8")
        save_json_file(classifications, docs_dir / "classification.json")
        save_json_file(log, self.config.processing_log_path)
