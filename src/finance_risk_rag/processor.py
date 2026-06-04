"""
Finance-Risk-RAG 文档处理器
==========================

负责 PDF 文档的文本提取与自动分类。
支持增量处理、OCR 优化和基于 LLM 的文档分类。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from .config import get_config
from .llm import LLMClientWrapper
from .utils import ensure_dirs, get_file_hash, load_json_file, save_json_file, setup_logger

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器类"""

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        初始化文档处理器

        Args:
            config: 配置对象
        """
        self._config = config or get_config()
        self._logger = setup_logger("document_processor", "logs/processor.log")

        # 初始化 LLM 客户端用于分类
        self._llm_client = LLMClientWrapper()

        # 配置 Tesseract
        if self._config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd

        # 确保目录存在
        ensure_dirs(self._config.cache_dir, self._config.docs_dir)

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        OCR 图像超优化

        Args:
            image: PIL 图像对象

        Returns:
            优化后的图像
        """
        # 转换为灰度图
        image = image.convert("L")
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
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def classify_document(self, text_sample: str) -> Dict[str, Any]:
        """
        使用 AI 自动分类文档

        Args:
            text_sample: 文本样本

        Returns:
            分类结果字典
        """
        if not self._llm_client.is_available:
            return {"type": "未知", "confidence": 0.0, "reason": "LLM 不可用"}

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
            content = self._llm_client.call(
                messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
            # 尝试解析 JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                return json.loads(json_str)
            return {"type": "未知", "confidence": 0.0, "reason": "无法解析 AI 输出"}
        except Exception as e:
            self._logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": str(e)}

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, int]:
        """
        从 PDF 提取文本，必要时使用 OCR

        Args:
            pdf_path: PDF 文件路径

        Returns:
            (提取的文本, OCR 页数)
        """
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    # 如果提取文本太少，尝试 OCR
                    if page_text and len(page_text.strip()) > 50:
                        text += f"\n--- Page {i+1} (Text) ---\n{page_text}"
                    else:
                        self._logger.info(f"页面 {i+1} 文本不足，启动 OCR...")
                        img = page.to_image(resolution=self._config.ocr_dpi).original
                        img = self.optimize_image_for_ocr(img)
                        ocr_text = pytesseract.image_to_string(
                            img, lang=self._config.ocr_languages, config="--oem 1 --psm 3"
                        )
                        text += f"\n--- Page {i+1} (OCR) ---\n{ocr_text}"
                        ocr_pages += 1

            return text, ocr_pages
        except Exception as e:
            self._logger.error(f"处理 PDF 失败 {pdf_path}: {e}")
            raise e

    def process_batch(self, input_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        批量处理文档

        Args:
            input_dir: 输入目录

        Returns:
            处理统计信息
        """
        input_dir = input_dir or self._config.docs_dir
        pdf_files = list(input_dir.glob("*.pdf"))

        log = load_json_file(self._config.processing_log_path, default={})
        all_text = ""
        classifications = {}
        stats = {"processed": 0, "skipped": 0, "total_ocr_pages": 0}

        for pdf_path in pdf_files:
            filename = pdf_path.name
            file_hash = get_file_hash(pdf_path)
            txt_path = pdf_path.with_suffix(".txt")

            # 检查增量处理
            cached = log.get(filename, {})
            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self._config.ocr_version
                and txt_path.exists()
            ):

                self._logger.info(f"跳过已处理文件: {filename}")
                all_text += txt_path.read_text(encoding="utf-8") + "\n\n" + "=" * 60 + "\n\n"
                classifications[filename] = cached.get("classification")
                stats["skipped"] += 1
                continue

            self._logger.info(f"正在处理: {filename}")
            try:
                text, ocr_count = self.extract_text_from_pdf(pdf_path)

                # AI 分类
                classification = self.classify_document(text[:3000])

                # 保存单个文本文件
                txt_path.write_text(text, encoding="utf-8")

                # 更新日志
                log[filename] = {
                    "hash": file_hash,
                    "ocr_version": self._config.ocr_version,
                    "processed_at": datetime.now().isoformat(),
                    "classification": classification,
                    "ocr_pages": ocr_count,
                }

                all_text += text + "\n\n" + "=" * 60 + "\n\n"
                classifications[filename] = classification
                stats["processed"] += 1
                stats["total_ocr_pages"] += ocr_count

            except Exception as e:
                self._logger.error(f"处理文件 {filename} 失败: {e}")

        # 保存合并文本和分类结果
        (input_dir / "all_extracted.txt").write_text(all_text, encoding="utf-8")
        save_json_file(classifications, input_dir / "classification.json")
        save_json_file(log, self._config.processing_log_path)

        self._logger.info(f"批量处理完成: {stats}")
        return stats


def main() -> None:
    """命令行入口"""
    processor = DocumentProcessor()
    stats = processor.process_batch()
    print(f"处理完成！统计: {stats}")


if __name__ == "__main__":
    main()
