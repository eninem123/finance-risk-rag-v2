"""
Finance-Risk-RAG 风险分析服务层
==============================

编排文档处理、实体提取和 RAG 查询流程，生成综合风险报告。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import load_json_file, save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风险分析综合服务类"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.processor = DocumentProcessor(self.config)
        self.extractor = EntityExtractionPipeline(self.config)
        self.engine = RAGEngine(self.config)

    def analyze_directory(self, docs_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        对指定目录进行全流程分析：OCR -> 分类 -> 实体提取 -> 建立索引
        """
        docs_dir = docs_dir or self.config.docs_dir
        logger.info(f"Starting full analysis for directory: {docs_dir}")

        # 1. 文档处理 (OCR + 分类)
        self.processor.process_directory(docs_dir)

        # 2. 实体提取
        all_extracted_path = docs_dir / "all_extracted.txt"
        extraction_result = self.extractor.process(all_extracted_path)

        # 3. 建立 RAG 索引
        self.engine.build_index()

        # 4. 生成初步报告数据
        report_data = {
            "directory": str(docs_dir),
            "summary": extraction_result.to_dict(),
            "status": "completed"
        }

        report_path = docs_dir / "risk_report.json"
        save_json_file(report_data, report_path)
        logger.info(f"Full analysis completed. Report saved to {report_path}")

        return report_data

    def generate_markdown_report(self, report_data: Dict[str, Any], output_path: Path):
        """生成易读的 Markdown 格式风险报告"""
        summary = report_data["summary"]

        lines = [
            "# 🏦 财务风险分析报告",
            f"\n**分析时间**: {summary.get('extracted_at', 'N/A')}",
            f"**风险等级**: {summary.get('risk_level', '未知')}",
            f"**总风险评分**: {summary.get('total_risk_score', 0)}",
            f"**模型版本**: {summary.get('model_version', 'N/A')}",
            "\n---",
            "\n## 🔍 识别到的风险实体",
            "| 类型 | 实体文本 | 风险分数 | 置信度 | 来源 |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]

        for entity in summary.get("entities", []):
            lines.append(
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']} | {entity['source']} |"
            )

        lines.append("\n\n## 💡 风险建议 (基于 RAG)")

        # 尝试使用 RAG 获取建议
        try:
            query = "根据识别到的风险点，给出专业的财务风险防范建议。"
            rag_res = self.engine.query(query)
            lines.append(rag_res.answer)
        except Exception as e:
            lines.append(f"获取 RAG 建议失败: {e}")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report generated at {output_path}")
