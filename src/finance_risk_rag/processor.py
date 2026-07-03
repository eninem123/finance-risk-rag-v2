"""
Finance-Risk-RAG 文档处理模块
============================
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import get_config
from .exceptions import OCRError
from .llm import LLMClientWrapper
from .models import ClassificationResult
from .utils import get_file_hash, load_json_file, save_json_file

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """处理 PDF 文档并提取文本，支持高级 OCR 预处理和智能分类"""

    def __init__(self, config=None, llm_client=None):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientWrapper()
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        高级图像预处理：
        1. 灰度化 -> 2. 自动对比度 -> 3. 中值滤波 -> 4. 锐化 -> 5. 二值化
        """
        # 1. 灰度化
        image = image.convert("L")

        # 2. 自动对比度优化
        image = ImageOps.autocontrast(image)

        # 3. 中值滤波去噪
        image = image.filter(ImageFilter.MedianFilter(size=3))

        # 4. 增强锐度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # 5. 自适应二值化 (模拟)
        image = image.point(lambda x: 0 if x < 150 else 255, "1")

        return image

    def classify_document(self, text_sample: str) -> ClassificationResult:
        """
        利用 LLM 对文档进行智能分类。
        """
        if not self.llm_client.is_available:
            return ClassificationResult(type="未知", confidence=0.0, reason="LLM unavailable")

        prompt = f"""
请分析以下财务文档的内容片段，并将其归类为以下之一。输出严格的 JSON 格式。

文档片段：
{text_sample[:3000]}

【候选类别】
1. 审计报告
2. 行业报告
3. 公司研究报告
4. 招股说明书
5. 财报
6. 其他

【输出示例】
{{"type": "财报", "confidence": 0.95, "reason": "文中包含资产负债表和利润表关键字。"}}
"""
        try:
            response = self.llm_client.chat([{"role": "user", "content": prompt}], temperature=0.1)

            # 解析 JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                data = json.loads(response[start:end])
                return ClassificationResult(
                    type=data.get("type", "其他"),
                    confidence=data.get("confidence", 0.0),
                    reason=data.get("reason", ""),
                )
            return ClassificationResult(type="未知", confidence=0.0, reason="JSON parse failed")
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(type="未知", confidence=0.0, reason=str(e))

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """
        从 PDF 提取文本，如果文本内容过少则触发 OCR。
        """
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    # 启发式判断：如果文本量太少，可能是扫描件
                    if page_text and len(page_text.strip()) > 100:
                        text += f"\n--- Page {i+1} (Text) ---\n{page_text}"
                    else:
                        # 触发 OCR
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

    def process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """处理单个 PDF，支持缓存检查"""
        file_hash = get_file_hash(pdf_path)
        log = load_json_file(self.config.processing_log_path)
        cached = log.get(pdf_path.name, {})
        txt_path = pdf_path.with_suffix(".txt")

        if (
            cached.get("hash") == file_hash
            and cached.get("ocr_version") == self.config.ocr_version
            and txt_path.exists()
        ):
            logger.info(f"Using cached result for {pdf_path.name}")
            text = txt_path.read_text(encoding="utf-8")
            classification_dict = cached.get("classification", {"type": "未知", "confidence": 0.0})
            ocr_count = cached.get("ocr_pages", 0)
        else:
            logger.info(f"Processing {pdf_path.name}")
            text, ocr_count = self.extract_text_from_pdf(pdf_path)
            txt_path.write_text(text, encoding="utf-8")

            # 分类逻辑
            classification = self.classify_document(text)
            classification_dict = classification.to_dict()

        return {
            "name": pdf_path.name,
            "text": text,
            "classification": classification_dict,
            "hash": file_hash,
            "ocr_pages": ocr_count,
            "ocr_version": self.config.ocr_version,
        }

    def process_directory(self, docs_dir: Optional[Path] = None, max_workers: int = 4):
        """批量处理目录"""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))
        log = load_json_file(self.config.processing_log_path)

        all_text = ""
        classifications = {}
        results = []

        if not pdf_files:
            logger.info("No PDF files found.")
            return

        if max_workers <= 1:
            for pdf in pdf_files:
                try:
                    results.append(self.process_single_pdf(pdf))
                except Exception as e:
                    logger.error(f"Failed to process {pdf.name}: {e}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pdf = {
                    executor.submit(self.process_single_pdf, pdf): pdf for pdf in pdf_files
                }
                for future in as_completed(future_to_pdf):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        logger.error(f"Parallel processing error: {e}")

        results.sort(key=lambda x: x["name"])

        for res in results:
            name = res["name"]
            all_text += res["text"] + "\n\n" + "=" * 60 + "\n\n"
            classifications[name] = res["classification"]
            log[name] = {
                "hash": res["hash"],
                "ocr_version": res["ocr_version"],
                "classification": res["classification"],
                "ocr_pages": res["ocr_pages"],
            }

        (docs_dir / "all_extracted.txt").write_text(all_text, encoding="utf-8")
        save_json_file(classifications, docs_dir / "classification.json")
        save_json_file(log, self.config.processing_log_path)
