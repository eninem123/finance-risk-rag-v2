"""
Finance-Risk-RAG 文档处理器
============================
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .engine import LLMClientWrapper
from .utils import ensure_dirs, setup_logger

logger = setup_logger("document_processor", "logs/processor_optimized.log")


class DocumentProcessor:
    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config or get_config()
        self.llm = LLMClientWrapper()

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        ensure_dirs(self.config.cache_dir, self.config.docs_dir)

    def _optimize_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))

        b_enhancer = ImageEnhance.Brightness(image)
        image = b_enhancer.enhance(1.2)

        c_enhancer = ImageEnhance.Contrast(image)
        image = c_enhancer.enhance(2.5)

        image = image.filter(ImageFilter.SHARPEN)
        return image.point(lambda x: 0 if x < 140 else 255, "1")

    def classify_document(self, text_sample: str) -> Dict[str, Any]:
        """使用 LLM 自动分类文档"""
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
- 格式示例：
{{
  "type": "行业报告",
  "confidence": 0.93,
  "reason": "文本主要为行业市场分析和趋势预测"
}}
"""
        messages = [{"role": "user", "content": prompt}]
        try:
            content = self.llm.call(messages, temperature=0.2)
            # 兼容模型多余文字
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                return cast(Dict[str, Any], json.loads(content[start:end]))
            return {"type": "未知", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {"type": "未知", "confidence": 0.0}

    def process_pdf(self, pdf_path: Path) -> str:
        text = f"# Source: {pdf_path.name}\n\n"
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    text += f"\n--- Page {i + 1} ---\n{page_text}"
                else:
                    img = page.to_image(resolution=self.config.ocr_dpi).original
                    img = self._optimize_image(img)
                    ocr_text = pytesseract.image_to_string(img, lang=self.config.ocr_languages)
                    text += f"\n--- Page {i + 1} (OCR) ---\n{ocr_text}"

        return text

    def batch_process(self) -> Dict[str, Any]:
        stats = {"processed": 0, "skipped": 0}
        classifications = {}
        for pdf_file in self.config.docs_dir.glob("*.pdf"):
            txt_path = pdf_file.with_suffix(".txt")
            if txt_path.exists():
                stats["skipped"] += 1
                continue

            logger.info(f"Processing {pdf_file.name}")
            text = self.process_pdf(pdf_file)
            txt_path.write_text(text, encoding="utf-8")

            # Perform classification
            classification = self.classify_document(text[:3000])
            classifications[pdf_file.name] = classification

            stats["processed"] += 1

        return {"stats": stats, "classifications": classifications}
