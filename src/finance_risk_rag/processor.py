import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from finance_risk_rag.config import get_config
from finance_risk_rag.exceptions import OCRError
from finance_risk_rag.llm import LLMClientWrapper
from finance_risk_rag.utils import get_file_hash, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理类 (OCR + 分类)"""

    def __init__(self) -> None:
        self.config = get_config()
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        self.llm_client = LLMClientWrapper()

    def _optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR 图像优化"""
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        bright_enhancer = ImageEnhance.Brightness(image)
        image = bright_enhancer.enhance(1.2)
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(2.5)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

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
        try:
            content = self.llm_client.ask(
                query=prompt, context="", system_prompt="你是一个文档分类专家，只返回 JSON。"
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": str(e)}

    def extract_text_from_pdf(
        self, pdf_path: Path, output_txt: Path
    ) -> Tuple[str, Dict[str, Any], int]:
        """从 PDF 提取文本，必要时使用 OCR"""
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
                        img = self._optimize_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(
                            img, lang=self.config.ocr_languages, config="--oem 1 --psm 3"
                        )
                        text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                        ocr_pages += 1

            output_txt.write_text(text, encoding="utf-8")
            classification = self.classify_document(text[:3000])
            return text, classification, ocr_pages

        except Exception as e:
            logger.error(f"提取 PDF 文本失败 {pdf_path}: {e}")
            raise OCRError(f"提取 PDF 文本失败: {e}") from e

    def batch_process(self) -> Dict[str, Any]:
        """批量处理文档目录"""
        pdf_files = list(self.config.docs_dir.glob("*.pdf"))
        log = load_json_file(self.config.processing_log_path)
        all_text = ""
        classifications = {}
        total_ocr = 0
        processed = 0

        for pdf_path in pdf_files:
            filename = pdf_path.name
            file_hash = get_file_hash(pdf_path)
            txt_path = pdf_path.with_suffix(".txt")

            cached = log.get(filename, {})
            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self.config.ocr_version
                and txt_path.exists()
            ):
                logger.info(f"跳过: {filename} (已处理)")
                text = txt_path.read_text(encoding="utf-8")
                classification = cached["classification"]
            else:
                logger.info(f"处理: {filename}")
                text, classification, ocr_count = self.extract_text_from_pdf(pdf_path, txt_path)
                total_ocr += ocr_count
                processed += 1

                log[filename] = {
                    "hash": file_hash,
                    "ocr_version": self.config.ocr_version,
                    "processed_at": datetime.now().isoformat(),
                    "classification": classification,
                    "ocr_pages": ocr_count,
                }

            all_text += text + "\n\n" + "=" * 60 + "\n\n"
            classifications[filename] = classification

        (self.config.docs_dir / "all_extracted.txt").write_text(all_text, encoding="utf-8")
        save_json_file(classifications, self.config.docs_dir / "classification.json")
        save_json_file(log, self.config.processing_log_path)

        return {
            "processed": processed,
            "skipped": len(pdf_files) - processed,
            "total_ocr_pages": total_ocr,
        }
