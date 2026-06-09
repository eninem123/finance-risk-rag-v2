"""
Finance-Risk-RAG 文档处理模块
============================

负责从PDF文件中提取文本，支持OCR识别和文档分类。

功能:
    - PDF文本提取
    - 图像增强与OCR优化
    - 文档自动分类
    - 增量处理与缓存管理
"""

import glob
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import pytesseract
from openai import OpenAI
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .exceptions import OCRError
from .utils import setup_logger


class DocumentProcessor:
    """文档处理与OCR引擎"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = setup_logger("document_processor", self.config.log_dir / "extract_text.log")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url
        )

        # 设置 Tesseract 路径
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        self.log_file = self.config.processing_log_path
        self.cache_dir = self.config.cache_dir
        self.config.ensure_directories()

    def get_file_hash(self, pdf_path: Path) -> str:
        """计算文件 MD5"""
        with open(pdf_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def load_log(self) -> Dict:
        """加载处理日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"加载日志失败: {e}")
        return {}

    def save_log(self, log: Dict) -> None:
        """保存处理日志"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    def classify_document_with_ai(self, text_sample: str) -> Dict:
        """用 AI 自动分类文档"""
        if not self.config.llm_api_key:
            return {"type": "未知", "confidence": 0.0, "reason": "未配置 API 密钥"}

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
                temperature=0.2
            )
            content = response.choices[0].message.content.strip()
            # 兼容模型多余文字
            start = content.find("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
            return json.loads(json_str)
        except Exception as e:
            self.logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": f"AI 分类异常: {e}"}

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR 超优化"""
        # 转换为灰度图
        image = image.convert('L')
        # 中值滤波去噪
        image = image.filter(ImageFilter.MedianFilter(size=3))
        # 增强亮度
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        # 增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)
        # 锐化
        image = image.filter(ImageFilter.SHARPEN)
        # 二值化
        image = image.point(lambda x: 0 if x < 140 else 255, '1')
        return image

    def extract_text_from_pdf(self, pdf_path: Path, output_txt: Path) -> Tuple[str, Dict, int]:
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
                        # 需要 OCR
                        img = page.to_image(resolution=self.config.ocr_dpi).original
                        img = self.optimize_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(
                            img,
                            lang=self.config.ocr_languages,
                            config='--oem 1 --psm 3'
                        )
                        text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                        ocr_pages += 1
                        self.logger.info(f"   [OCR] {pdf_path.name} Page {i+1} 已识别")
        except Exception as e:
            self.logger.error(f"PDF 处理失败 {pdf_path}: {e}")
            raise OCRError(f"无法从 PDF 提取文本: {e}")

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(text)

        self.logger.info(f"提取完成: {output_txt} (OCR 页数: {ocr_pages})")

        sample = text[:3000]
        classification = self.classify_document_with_ai(sample)
        self.logger.info(f"   [分类] {classification['type']} (置信度: {classification['confidence']:.2f})")

        return text, classification, ocr_pages

    def batch_process(self) -> None:
        """批量处理文档目录"""
        pdf_files = list(self.config.docs_dir.glob("*.pdf"))
        log = self.load_log()
        all_text = ""
        classifications = {}
        total_ocr = 0
        processed = 0

        for pdf_path in pdf_files:
            filename = pdf_path.name
            file_hash = self.get_file_hash(pdf_path)
            txt_path = self.config.docs_dir / (pdf_path.stem + ".txt")

            # 检查增量处理
            cached = log.get(filename, {})
            if (cached.get("hash") == file_hash and
                cached.get("ocr_version") == self.config.ocr_version and
                txt_path.exists()):
                self.logger.info(f"跳过: {filename} (已处理，版本 {self.config.ocr_version})")
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                classification = cached["classification"]
            else:
                self.logger.info(f"处理: {filename}")
                try:
                    text, classification, ocr_count = self.extract_text_from_pdf(pdf_path, txt_path)
                    total_ocr += ocr_count
                    processed += 1

                    # 更新日志
                    log[filename] = {
                        "hash": file_hash,
                        "ocr_version": self.config.ocr_version,
                        "processed_at": datetime.now().isoformat(),
                        "classification": classification,
                        "ocr_pages": ocr_count
                    }
                except Exception as e:
                    self.logger.error(f"处理 {filename} 失败: {e}")
                    continue

            all_text += text + "\n\n" + "="*60 + "\n\n"
            classifications[filename] = classification

        # 保存汇总结果
        all_extracted_path = self.config.docs_dir / "all_extracted.txt"
        with open(all_extracted_path, "w", encoding="utf-8") as f:
            f.write(all_text)

        classification_json_path = self.config.docs_dir / "classification.json"
        with open(classification_json_path, "w", encoding="utf-8") as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)

        self.save_log(log)

        self.logger.info(f"\n批量处理完成！本次处理 {processed} 个文件，跳过 {len(pdf_files)-processed} 个")
        self.logger.info(f"   OCR 总页数: {total_ocr}，版本: {self.config.ocr_version}")


def main():
    processor = DocumentProcessor()
    processor.batch_process()


if __name__ == "__main__":
    main()
