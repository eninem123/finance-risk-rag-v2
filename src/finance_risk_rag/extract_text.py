"""
Finance-Risk-RAG 文档处理模块 (OCR + 分类)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pdfplumber
import pytesseract
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from finance_risk_rag.config import get_config
from finance_risk_rag.utils import get_file_hash, setup_logger


class DocumentProcessor:
    """
    文档处理器，负责从 PDF 提取文本（支持 OCR）并进行 AI 分类。
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config or get_config()
        self.logger = setup_logger("document_processor")
        self.client = None
        if self.config.llm_api_key:
            self.client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_base_url)

        # 设置 Tesseract 路径
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        优化图像以提高 OCR 准确率
        """
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))

        enhancer_b = ImageEnhance.Brightness(image)
        image = enhancer_b.enhance(1.2)

        enhancer_c = ImageEnhance.Contrast(image)
        image = enhancer_c.enhance(2.5)

        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def classify_with_ai(self, text_sample: str) -> Dict[str, Any]:
        """
        使用 AI 对文档进行分类
        """
        if not self.client:
            return {"type": "未知", "confidence": 0.0, "reason": "AI 客户端未配置"}

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
"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw_content = response.choices[0].message.content
            if raw_content is None:
                return {"type": "未知", "confidence": 0.0, "reason": "API 返回空内容"}
            content = raw_content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            self.logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": str(e)}

    def extract_text(self, pdf_path: Path) -> Tuple[str, Dict[str, Any], int]:
        """
        从单个 PDF 提取文本
        """
        self.logger.info(f"正在处理: {pdf_path}")
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text is not None and len(page_text.strip()) > 50:
                    text += f"\n--- Page {i+1} (Text) ---\n{page_text}"
                else:
                    img = page.to_image(resolution=self.config.ocr_dpi).original
                    img = self.optimize_image_for_ocr(img)
                    ocr_text = pytesseract.image_to_string(
                        img, lang=self.config.ocr_languages, config="--oem 1 --psm 3"
                    )
                    text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                    ocr_pages += 1

        classification = self.classify_with_ai(text[:3000])
        return text, classification, ocr_pages

    def batch_process(self) -> None:
        """
        批量处理文档目录下的 PDF 文件
        """
        self.config.ensure_directories()
        pdf_files = list(self.config.docs_dir.glob("*.pdf"))
        log = self._load_log()
        all_extracted_text = ""
        classifications = {}

        for pdf_path in pdf_files:
            file_hash = get_file_hash(pdf_path)
            txt_path = pdf_path.with_suffix(".txt")

            cached = log.get(pdf_path.name, {})
            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self.config.ocr_version
                and txt_path.exists()
            ):
                self.logger.info(f"跳过已处理文件: {pdf_path.name}")
                text = txt_path.read_text(encoding="utf-8")
                classification = cached["classification"]
            else:
                text, classification, ocr_count = self.extract_text(pdf_path)
                txt_path.write_text(text, encoding="utf-8")
                log[pdf_path.name] = {
                    "hash": file_hash,
                    "ocr_version": self.config.ocr_version,
                    "processed_at": datetime.now().isoformat(),
                    "classification": classification,
                    "ocr_pages": ocr_count,
                }

            all_extracted_text += text + "\n\n" + "=" * 60 + "\n\n"
            classifications[pdf_path.name] = classification

        # 保存汇总结果
        (self.config.docs_dir / "all_extracted.txt").write_text(
            all_extracted_text, encoding="utf-8"
        )
        (self.config.docs_dir / "classification.json").write_text(
            json.dumps(classifications, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._save_log(log)

    def _load_log(self) -> Dict[str, Any]:
        log_path = self.config.cache_dir / "processing_log.json"
        if log_path.exists():
            return json.loads(log_path.read_text(encoding="utf-8"))
        return {}

    def _save_log(self, log: Dict[str, Any]) -> None:
        log_path = self.config.cache_dir / "processing_log.json"
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
