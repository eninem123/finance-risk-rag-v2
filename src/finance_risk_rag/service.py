"""
Finance-Risk-RAG 编排服务模块
============================

协调文档处理、实体提取和 RAG 查询的顶层服务。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import setup_logger

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风险分析编排服务"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
        self.extractor = EntityExtractionPipeline(self.config)
        self.engine = RAGEngine(self.config)
        setup_logger("finance_risk_rag")

    def run_full_analysis(self, pdf_path: Path) -> Dict[str, Any]:
        """对单个 PDF 执行全流程分析"""
        logger.info(f"Starting full analysis for {pdf_path.name}")

        # 1. 文档处理与 OCR
        proc_result = self.processor.process_single_pdf(pdf_path)
        text = proc_result["text"]

        # 2. 实体提取
        ext_result = self.extractor.process(text)

        # 3. 自动生成摘要问题
        summary_query = "请总结该文档的主要财务风险点。"
        rag_result = self.engine.query(summary_query)

        return {
            "filename": pdf_path.name,
            "classification": proc_result["classification"],
            "extraction": ext_result.to_dict(),
            "summary": rag_result.answer,
            "ocr_pages": proc_result["ocr_pages"],
        }

    def generate_batch_report(self, docs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """生成整个目录的批量风险报告"""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))

        # 确保索引已构建
        logger.info("Ensuring RAG index is up to date...")
        self.processor.process_directory(docs_dir)
        self.engine.build_index()

        reports = []
        for pdf in pdf_files:
            try:
                report = self.run_full_analysis(pdf)
                reports.append(report)
            except Exception as e:
                logger.error(f"Analysis failed for {pdf.name}: {e}")

        return reports
