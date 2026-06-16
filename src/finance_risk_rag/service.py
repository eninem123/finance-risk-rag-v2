"""
Finance-Risk-RAG 风险分析服务 (Orchestration Layer)
==================================================
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """整合 OCR、实体提取和 RAG 的业务服务"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
        self.pipeline = EntityExtractionPipeline(self.config)
        self.engine = RAGEngine(self.config)

    def analyze_document(self, pdf_path: Path) -> Dict:
        """全流程分析单个文档"""
        # 1. OCR 与 文本提取
        process_res = self.processor.process_single_pdf(pdf_path)
        text = process_res["text"]

        # 2. 实体提取
        extraction_res = self.pipeline.process(text)

        # 3. 自动生成摘要与风险点 (RAG)
        # 先确保该文档已入库
        txt_path = pdf_path.with_suffix(".txt")
        self.engine.add_documents([txt_path])

        summary_query = "请总结这份文档的主要财务状况和潜在风险点。"
        summary_res = self.engine.query(summary_query)

        report = {
            "document_name": pdf_path.name,
            "classification": process_res["classification"],
            "risk_analysis": extraction_res.to_dict(),
            "ai_summary": summary_res.answer,
            "sources": summary_res.sources,
        }

        return report

    def batch_analyze(self, docs_dir: Optional[Path] = None) -> List[Dict]:
        """批量处理目录下的文档"""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))

        reports = []
        for pdf in pdf_files:
            try:
                report = self.analyze_document(pdf)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to analyze {pdf}: {e}")

        # 保存汇总报告
        save_json_file(reports, docs_dir / "comprehensive_risk_report.json")
        return reports
