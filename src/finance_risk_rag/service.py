"""
Finance-Risk-RAG 业务服务编排模块
================================
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """
    风控分析服务编排器，协调文档处理、实体提取和 RAG 查询。
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
        self.pipeline = EntityExtractionPipeline(self.config)
        self.engine = RAGEngine(self.config)

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        全流程分析单个文档：OCR -> 分类 -> 实体提取 -> 报告生成
        """
        logger.info(f"Starting analysis for {pdf_path.name}")

        # 1. 文档处理 (OCR + 分类)
        proc_res = self.processor.process_single_pdf(pdf_path)
        text = proc_res["text"]
        classification = proc_res["classification"]

        # 2. 实体提取
        extraction_res = self.pipeline.process(text)

        # 3. 构建结果
        report = {
            "document_name": pdf_path.name,
            "classification": classification,
            "risk_analysis": extraction_res.to_dict(),
            "summary": self._generate_summary(extraction_res),
        }

        # 4. 自动加入 RAG 索引
        txt_path = pdf_path.with_suffix(".txt")
        if txt_path.exists():
            self.engine.add_documents([txt_path])

        return report

    def _generate_summary(self, result) -> str:
        """根据提取结果生成简要总结"""
        if not result.entities:
            return "未发现显著风险实体。"

        return (
            f"发现 {len(result.entities)} 个风险实体。"
            f"系统评估风险等级为: {result.risk_level} (得分: {result.total_risk_score})。"
        )

    def run_batch_analysis(self, docs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """批量分析目录下的所有 PDF"""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))

        reports = []
        for pdf in pdf_files:
            try:
                report = self.analyze_document(pdf)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to analyze {pdf.name}: {e}")

        # 保存汇总报告
        save_json_file(reports, docs_dir / "batch_risk_report.json")
        return reports

    def query_risk(self, question: str) -> Dict[str, Any]:
        """执行 RAG 风险问答"""
        result = self.engine.query(question)
        return {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
        }
