"""
Finance-Risk-RAG OCR 文本提取模块
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pdfplumber
import pytesseract
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from finance_risk_rag.config import get_config
from finance_risk_rag.utils import (
    get_file_hash,
    load_json_file,
    save_json_file,
    setup_logger,
)


class DocumentProcessor:
    """文档处理类，负责 OCR 识别与分类"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("document_processor", str(self.config.log_dir / "processor.log"))

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        self.client = None
        if self.config.llm_api_key:
            self.client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_base_url)

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """图像预处理优化"""
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def classify_document(self, text_sample: str) -> Dict:
        """使用 LLM 自动分类文档"""
        if not self.client:
            return {"type": "未知", "confidence": 0.0, "reason": "LLM 客户端未配置"}

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
            response = self.client.chat.completions.create(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = response.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            self.logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": str(e)}

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict, int]:
        """从 PDF 提取文本，必要时使用 OCR"""
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0

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
                    self.logger.info(f"   [OCR] {pdf_path.name} Page {i+1} 已识别")

        classification = self.classify_document(text[:3000])
        return text, classification, ocr_pages

    def batch_process(self, input_dir: str):
        """批量处理目录下的 PDF 文件"""
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("*.pdf"))

        log_path = self.config.processing_log_path
        log = load_json_file(log_path, {})

        all_texts = []
        classifications = {}

        for pdf_file in pdf_files:
            file_hash = get_file_hash(pdf_file)
            cached = log.get(pdf_file.name, {})

            txt_path = pdf_file.with_suffix(".txt")

            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self.config.ocr_version
                and txt_path.exists()
            ):
                self.logger.info(f"跳过已处理文件: {pdf_file.name}")
                text = txt_path.read_text(encoding="utf-8")
                classification = cached["classification"]
            else:
                self.logger.info(f"正在处理: {pdf_file.name}")
                text, classification, ocr_count = self.extract_text_from_pdf(pdf_file)
                txt_path.write_text(text, encoding="utf-8")

                log[pdf_file.name] = {
                    "hash": file_hash,
                    "ocr_version": self.config.ocr_version,
                    "processed_at": datetime.now().isoformat(),
                    "classification": classification,
                    "ocr_pages": ocr_count,
                }

            all_texts.append(text)
            classifications[pdf_file.name] = classification

        # 合并保存
        combined_txt = self.config.docs_dir / "all_extracted.txt"
        separator = "\n\n" + "=" * 60 + "\n\n"
        combined_txt.write_text(separator.join(all_texts), encoding="utf-8")

        save_json_file(classifications, self.config.docs_dir / "classification.json")
        save_json_file(log, log_path)
        self.logger.info("批量处理完成")
