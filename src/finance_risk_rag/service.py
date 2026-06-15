"""
Finance-Risk-RAG 风险分析服务
============================

编排层：协调文档处理、实体提取和 RAG 问答。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """
    风险分析服务：提供一站式文档风控解决方案。
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
        self.extractor = EntityExtractionPipeline(self.config)
        self.engine = RAGEngine(self.config)

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """
        全流程分析单个文档。
        """
        logger.info(f"Analyzing document: {pdf_path.name}")

        # 1. 文档处理与 OCR
        proc_res = self.processor.process_single_pdf(pdf_path)
        text = proc_res["text"]

        # 2. 实体提取
        ext_res = self.extractor.process(text)

        # 3. 自动生成摘要与风险点（通过 RAG/LLM）
        # 这里我们可以直接使用 engine 里的 llm 或者通过 RAG 查询
        # 先构建/更新索引（如果需要，或者针对单文档使用内存向量库）
        # 简单起见，这里假设已经有索引或者我们只做文本分析

        summary_query = "请总结这份文档的主要财务状况，并列出关键的潜在风险点。"
        rag_res = self.engine.query(summary_query)

        report = {
            "document_name": pdf_path.name,
            "classification": proc_res["classification"],
            "risk_assessment": {
                "level": ext_res.risk_level,
                "score": ext_res.total_risk_score,
                "entities": [e.to_dict() for e in ext_res.entities]
            },
            "ai_analysis": {
                "summary": rag_res.answer,
                "sources": rag_res.sources
            },
            "metadata": {
                "ocr_pages": proc_res["ocr_pages"],
                "file_hash": proc_res["hash"]
            }
        }

        return report

    def generate_report_markdown(self, report: Dict[str, Any]) -> str:
        """
        生成 Markdown 格式的风险报告。
        """
        md = f"# 财务风险分析报告: {report['document_name']}\n\n"
        md += f"## 1. 文档概览\n"
        md += f"- **文档类型**: {report['classification']['type']}\n"
        md += f"- **分类置信度**: {report['classification']['confidence']:.2f}\n"
        md += f"- **OCR 页数**: {report['metadata']['ocr_pages']}\n\n"

        md += f"## 2. 风险评估\n"
        md += f"- **综合风险等级**: **{report['risk_assessment']['level']}**\n"
        md += f"- **量化风险评分**: {report['risk_assessment']['score']}\n\n"

        md += "### 2.1 识别出的关键风险实体\n"
        if not report['risk_assessment']['entities']:
            md += "未识别到明显的风险实体。\n"
        else:
            md += "| 类型 | 实体 | 分数 | 来源 | 上下文 |\n"
            md += "| --- | --- | --- | --- | --- |\n"
            for e in report['risk_assessment']['entities'][:20]: # 限制前20个
                ctx = e['context'][:50] + "..." if len(e['context']) > 50 else e['context']
                md += f"| {e['type']} | {e['text']} | {e['risk_score']} | {e['source']} | {ctx} |\n"

        md += f"\n## 3. AI 深度分析\n"
        md += f"{report['ai_analysis']['summary']}\n\n"

        md += "---\n*报告由 Finance-Risk-RAG 系统自动生成*"
        return md

    def process_and_report_directory(self, input_dir: Path, output_dir: Path):
        """
        处理目录下所有文档并生成汇总报告。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = list(input_dir.glob("*.pdf"))

        all_reports = []
        for pdf in pdf_files:
            try:
                report = self.analyze_document(pdf)
                all_reports.append(report)

                # 保存单个报告
                report_name = pdf.stem + "_report"
                save_json_file(report, output_dir / f"{report_name}.json")
                (output_dir / f"{report_name}.md").write_text(
                    self.generate_report_markdown(report), encoding="utf-8"
                )
            except Exception as e:
                logger.error(f"Failed to analyze {pdf.name}: {e}")

        # 保存汇总
        save_json_file(all_reports, output_dir / "summary_reports.json")
        logger.info(f"Successfully processed {len(all_reports)} documents.")
