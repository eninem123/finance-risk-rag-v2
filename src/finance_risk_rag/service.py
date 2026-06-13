"""
Finance-Risk-RAG 业务服务层
==========================

协调文档处理、实体提取和风险评估，生成综合报告。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, get_config
from .extractor import EntityExtractionPipeline
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风险分析编排服务"""

    def __init__(
        self,
        config: Optional[Config] = None,
        processor: Optional[DocumentProcessor] = None,
        extractor: Optional[EntityExtractionPipeline] = None,
    ):
        self.config = config or get_config()
        self.processor = processor or DocumentProcessor(self.config)
        self.extractor = extractor or EntityExtractionPipeline(self.config)

    def run_full_analysis(self, pdf_path: Path) -> Dict[str, Any]:
        """
        执行完整的分析流程：OCR -> 分类 -> 实体提取 -> 风险评估
        """
        logger.info(f"Starting full analysis for {pdf_path.name}")

        # 1. 文档处理 (OCR + 分类)
        proc_result = self.processor.process_single_pdf(pdf_path)
        text = proc_result["text"]
        classification = proc_result["classification"]

        # 2. 实体提取
        extraction_result = self.extractor.process(text)

        # 3. 组合结果
        analysis_data = {
            "document_info": {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "hash": proc_result["hash"],
                "analyzed_at": datetime.now().isoformat(),
            },
            "classification": classification,
            "risk_analysis": extraction_result.to_dict(),
        }

        return analysis_data

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """
        根据分析数据生成 Markdown 格式的风险报告
        """
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        report = f"""# 财务风险分析报告: {doc_info['name']}

## 1. 基本信息
- **分析时间**: {doc_info['analyzed_at']}
- **文档类型**: {classification.get('type', '未知')} (置信度: {classification.get('confidence', 0.0):.2f})
- **分类依据**: {classification.get('reason', '无')}

## 2. 风险评估摘要
- **风险等级**: **{risk['risk_level']}**
- **量化评分**: {risk['total_risk_score']}
- **识别实体总数**: {risk['total_entities']}

## 3. 详细风险实体
| 类型 | 实体文本 | 风险分数 | 置信度 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.2f} | {entity['source']} |\n"
            )

        report += "\n## 4. 结论与建议\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "⚠️ **建议**: 该文档包含多项高风险因素，建议进行人工深度审计和加强现场尽调。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "💡 **建议**: 存在一定风险点，建议关注相关实体的背景情况，" "必要时要求补充资料。\n"
            )
        else:
            report += "✅ **建议**: 风险较低，可按正常流程处理。\n"

        if output_path:
            output_path.write_text(report, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")

            # 同时保存 JSON 原始数据
            json_path = output_path.with_suffix(".json")
            save_json_file(analysis_data, json_path)

        return report

    def process_batch(self, directory: Path) -> List[Dict[str, Any]]:
        """批量处理目录下的所有 PDF"""
        pdf_files = list(directory.glob("*.pdf"))
        results = []
        for pdf in pdf_files:
            try:
                res = self.run_full_analysis(pdf)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to analyze {pdf.name}: {e}")
        return results
