"""
Finance-Risk-RAG 文档处理模块
============================
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    def __init__(self, config=None, llm_client=None):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientWrapper(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
        )
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
你是一个专业的金融文档分类专家。请分析以下文本样本，并将其归类为最合适的金融文档类型。

文本样本（前3000字）：
{text_sample[:3000]}

【候选类别】
- 审计报告 (Audit Report): 包含会计师事务所意见、财务报表审计。
- 行业报告 (Industry Report): 对特定行业的发展、竞争格局进行的深度分析。
- 公司研究报告 (Company Research): 针对特定上市公司的基本面分析、估值与建议。
- 上市手册/招股书 (Prospectus): 包含公司历史、业务、风险因素、募集资金用途。
- 定期财报 (Financial Statement): 季度、半年度或年度财务数据披露。
- 其他 (Other): 不属于上述类别的文档。

【输出要求】
- 必须严格输出 JSON 格式。
- 包含字段：type (类别名称), confidence (置信度0-1), reason (简短理由)。
- 不要输出任何其他解释性文本。

示例：
{{"type": "审计报告", "confidence": 0.98, "reason": "文本中包含审计意见及资产负债表信息"}}
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

    def process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """处理单个 PDF 的方法"""
        file_hash = get_file_hash(pdf_path)
        log = load_json_file(self.config.processing_log_path)
        cached = log.get(pdf_path.name, {})
        txt_path = pdf_path.with_suffix(".txt")

        if (
            cached.get("hash") == file_hash
            and cached.get("ocr_version") == self.config.ocr_version
            and txt_path.exists()
        ):
            logger.info(f"Skipping {pdf_path.name} (cached)")
            text = txt_path.read_text(encoding="utf-8")
            classification_dict = cached.get("classification", {"type": "未知", "confidence": 0.0})
            ocr_count = cached.get("ocr_pages", 0)
        else:
            logger.info(f"Processing {pdf_path.name}")
            text, ocr_count = self.extract_text_from_pdf(pdf_path)
            txt_path.write_text(text, encoding="utf-8")
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
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))
        log = load_json_file(self.config.processing_log_path)

        all_text = ""
        classifications = {}
        results = []

        if not pdf_files:
            logger.info("No PDF files found to process.")
            return

        # 并行处理文件
        if max_workers <= 1:
            for pdf in pdf_files:
                try:
                    results.append(self.process_single_pdf(pdf))
                except Exception as e:
                    logger.error(f"Failed to process {pdf.name}: {e}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_pdf = {executor.submit(self.process_single_pdf, pdf): pdf for pdf in pdf_files}
                for future in as_completed(future_to_pdf):
                    try:
                        res = future.result()
                        results.append(res)
                    except Exception as e:
                        pdf = future_to_pdf[future]
                        logger.error(f"Failed to process {pdf.name}: {e}")

        # 聚合结果并更新日志 (保持某种程度的顺序以便合并)
        # Sort by name to keep all_extracted.txt consistent
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
