"""
Finance-Risk-RAG 文档处理模块
============================

负责从 PDF 文档中提取文本，支持原生文本提取和 OCR 识别。
集成 AI 自动文档分类功能。

作者: Finance-Risk-RAG Team
版本: 2.0.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from config import get_config
from rag_core import LLMClientWrapper
from utils import ensure_dirs, get_file_hash, setup_logger

# 配置日志
logger = setup_logger("document_processor", "logs/extract_text.log")


class DocumentProcessor:
    """文档处理类，负责文本提取和分类"""

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        初始化文档处理器

        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.llm_client = LLMClientWrapper(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
        )

        # 配置 Tesseract
        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        ensure_dirs(self.config.cache_dir, self.config.docs_dir, self.config.log_dir)

    def classify_document_with_ai(self, text_sample: str) -> Dict[str, Any]:
        """
        用 AI 自动分类文档

        Args:
            text_sample: 文本样本

        Returns:
            分类结果字典
        """
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
            content = self.llm_client.call(messages, temperature=0.2)
            # 兼容模型可能输出的多余文字
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = content[start:end]
                return json.loads(json_str)
            else:
                logger.error(f"无法从响应中解析 JSON: {content}")
                return {"type": "未知", "confidence": 0.0, "reason": "解析失败"}
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return {"type": "未知", "confidence": 0.0, "reason": f"API 调用失败: {e}"}

    def optimize_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        OCR 图像预处理优化

        Args:
            image: 原始图像

        Returns:
            处理后的图像
        """
        image = image.convert("L")  # 转灰度
        image = image.filter(ImageFilter.MedianFilter(size=3))

        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(1.2)

        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(2.5)

        image = image.filter(ImageFilter.SHARPEN)
        # 二值化
        image = image.point(lambda x: 0 if x < 140 else 255, "1")
        return image

    def extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any], int]:
        """
        从单份 PDF 中提取文本

        Args:
            pdf_path: PDF 文件路径

        Returns:
            (提取的文本, 分类结果, OCR 页数)
        """
        text = f"# 文件: {pdf_path.name}\n\n"
        ocr_pages = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    # 如果提取到的文本太少，则尝试 OCR
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
                        logger.info(f"   [OCR] {pdf_path.name} Page {i+1} 已识别")
        except Exception as e:
            logger.error(f"提取 PDF 失败 {pdf_path}: {e}")
            raise

        # 抽取样本文本进行分类
        sample = text[:3000]
        classification = self.classify_document_with_ai(sample)
        conf = classification.get("confidence", 0)
        logger.info(f"   [分类] {pdf_path.name}: {classification.get('type')} (置信度: {conf:.2f})")

        return text, classification, ocr_pages

    def batch_process(self, docs_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        批量处理目录下的文档

        Args:
            docs_dir: 文档目录

        Returns:
            批处理统计信息
        """
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))

        log_path = self.config.processing_log_path
        log = {}
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f)

        all_text = ""
        classifications = {}
        total_ocr_pages = 0
        processed_count = 0
        skipped_count = 0

        for pdf_path in pdf_files:
            filename = pdf_path.name
            file_hash = get_file_hash(pdf_path)
            txt_path = pdf_path.with_suffix(".txt")

            # 检查是否需要增量处理
            cached = log.get(filename, {})
            if (
                cached.get("hash") == file_hash
                and cached.get("ocr_version") == self.config.ocr_version
                and txt_path.exists()
            ):
                logger.info(f"跳过: {filename} (已处理，版本 {self.config.ocr_version})")
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                classification = cached.get("classification", {"type": "未知"})
                skipped_count += 1
            else:
                logger.info(f"处理: {filename}")
                try:
                    text, classification, ocr_count = self.extract_text_from_pdf(pdf_path)

                    # 保存单文件文本
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    total_ocr_pages += ocr_count
                    processed_count += 1

                    # 更新日志
                    log[filename] = {
                        "hash": file_hash,
                        "ocr_version": self.config.ocr_version,
                        "processed_at": datetime.now().isoformat(),
                        "classification": classification,
                        "ocr_pages": ocr_count,
                    }
                except Exception as e:
                    logger.error(f"处理 {filename} 时出错: {e}")
                    continue

            all_text += text + "\n\n" + "=" * 60 + "\n\n"
            classifications[filename] = classification

        # 保存汇总结果
        all_extracted_path = docs_dir / "all_extracted.txt"
        with open(all_extracted_path, "w", encoding="utf-8") as f:
            f.write(all_text)

        classification_path = docs_dir / "classification.json"
        with open(classification_path, "w", encoding="utf-8") as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

        stats = {
            "total_files": len(pdf_files),
            "processed": processed_count,
            "skipped": skipped_count,
            "total_ocr_pages": total_ocr_pages,
            "ocr_version": self.config.ocr_version,
        }
        logger.info(f"批量处理完成: {stats}")
        return stats


def main() -> None:
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 文档处理工具")
    parser.add_argument("--docs-dir", type=str, help="PDF 文档所在目录")

    args = parser.parse_args()

    config = get_config()
    docs_dir = Path(args.docs_dir) if args.docs_dir else config.docs_dir

    processor = DocumentProcessor(config)
    stats = processor.batch_process(docs_dir)

    print("\n" + "=" * 30)
    print("文档处理总结")
    print("=" * 30)
    print(f"总文件数: {stats['total_files']}")
    print(f"本次处理: {stats['processed']}")
    print(f"跳过处理: {stats['skipped']}")
    print(f"OCR 总页数: {stats['total_ocr_pages']}")
    print(f"OCR 版本: {stats['ocr_version']}")
    print(f"日志路径: {config.processing_log_path}")
    print("=" * 30)


if __name__ == "__main__":
    main()
