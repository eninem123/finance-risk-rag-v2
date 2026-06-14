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
from .models import ExtractionResult, QueryResult
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风控分析编排服务"""

    def __init__(self, config=None, llm_client=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config, llm_client=llm_client)
        self.extractor = EntityExtractionPipeline(self.config)
        self.rag_engine = RAGEngine(self.config, llm_client=llm_client)

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """对单个文档进行全流程分析"""
        logger.info(f"Starting analysis for {pdf_path}")

        # 1. 文档处理 (OCR + 分类)
        doc_info = self.processor.process_single_pdf(pdf_path)
        text = doc_info["text"]

        # 2. 实体提取
        extraction_result = self.extractor.process(text)

        # 3. 报告生成 (整合信息)
        report = {
            "document": {
                "name": pdf_path.name,
                "type": doc_info["classification"]["type"],
                "confidence": doc_info["classification"]["confidence"],
            },
            "risk_analysis": extraction_result.to_dict(),
            "summary": self._generate_summary(doc_info, extraction_result),
        }

        # 4. 可选：自动添加到 RAG 索引
        # self.rag_engine.add_documents([pdf_path.with_suffix(".txt")])

        return report

    def _generate_summary(self, doc_info: Dict[str, Any], extraction: ExtractionResult) -> str:
        """生成风险简述"""
        doc_type = doc_info["classification"]["type"]
        risk_level = extraction.risk_level
        entity_count = len(extraction.entities)

        summary = f"该文档被识别为 [{doc_type}]。经 AI 扫描，"
        if entity_count > 0:
            summary += f"共发现 {entity_count} 处潜在风险点，整体评估等级为 [{risk_level}]。"
        else:
            summary += "未发现明显预定义风险点，整体评估等级为 [低风险]。"

        return summary

    def run_query(self, question: str) -> QueryResult:
        """执行 RAG 查询"""
        return self.rag_engine.query(question)

    def generate_batch_report(self, docs_dir: Optional[Path] = None):
        """批量处理并生成汇总报告"""
        docs_dir = docs_dir or self.config.docs_dir
        pdf_files = list(docs_dir.glob("*.pdf"))

        all_reports = []
        for pdf in pdf_files:
            try:
                report = self.analyze_document(pdf)
                all_reports.append(report)
            except Exception as e:
                logger.error(f"Failed to analyze {pdf.name}: {e}")

        # 保存汇总报告
        report_path = docs_dir / "risk_report_batch.json"
        save_json_file(all_reports, report_path)
        logger.info(f"Batch report saved to {report_path}")
        return all_reports
