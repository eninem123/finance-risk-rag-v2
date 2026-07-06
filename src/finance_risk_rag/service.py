"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层，协调文档处理、实体提取和 RAG 引擎。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Config, get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .models import ExtractionResult
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风险分析服务编排类"""

    def __init__(
        self,
        config: Optional[Config] = None,
        processor: Optional[DocumentProcessor] = None,
        pipeline: Optional[EntityExtractionPipeline] = None,
        engine: Optional[RAGEngine] = None,
        extractor: Optional[EntityExtractionPipeline] = None,
    ):
        self.config = config or get_config()
        self._processor = processor
        self._pipeline = pipeline or extractor
        self._engine = engine

    @property
    def processor(self) -> DocumentProcessor:
        if self._processor is None:
            self._processor = DocumentProcessor(self.config)
        return self._processor

    @property
    def pipeline(self) -> EntityExtractionPipeline:
        if self._pipeline is None:
            self._pipeline = EntityExtractionPipeline(self.config)
        return self._pipeline

    @property
    def engine(self) -> RAGEngine:
        if self._engine is None:
            self._engine = RAGEngine(self.config)
        return self._engine

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API"""
        return self.pipeline

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """执行单文档完整分析：OCR -> 分类 -> 实体提取"""
        logger.info("Starting full analysis for %s", pdf_path.name)

        proc_result = self.processor.process_single_pdf(pdf_path)
        extraction_result = self.pipeline.process(proc_result["text"])

        return {
            "document_info": {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "hash": proc_result["hash"],
                "analyzed_at": datetime.now().isoformat(),
            },
            "classification": proc_result["classification"],
            "risk_analysis": extraction_result.to_dict(),
        }

    def run_full_analysis(
        self, input_path: Path
    ) -> Union[Dict[str, ExtractionResult], Dict[str, Any]]:
        """
        执行全流程分析：OCR -> 文档分类 -> 实体提取 -> RAG 索引构建。
        单 PDF 返回详细分析字典，目录返回文件名到提取结果的映射。
        """
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            analysis_data = self.analyze_document(input_path)

            txt_path = input_path.with_suffix(".txt")
            if txt_path.exists():
                self.engine.add_documents([txt_path])

            return analysis_data

        results: Dict[str, ExtractionResult] = {}

        if input_path.is_dir():
            self.processor.process_directory(input_path)
            txt_files = list(input_path.glob("*.txt"))
            for txt_file in txt_files:
                if txt_file.name == "all_extracted.txt":
                    continue
                extraction_res = self.pipeline.process(txt_file)
                results[txt_file.stem + ".pdf"] = extraction_res

            self.engine.build_index()

        return results

    def _generate_executive_summary(self, risk: Dict[str, Any]) -> str:
        """生成执行摘要（可选：接入 LLM 进一步总结）"""
        total = risk["total_entities"]
        level = risk["risk_level"]
        score = risk["total_risk_score"]

        if total == 0:
            return "本报告未发现显著的财务风险点。文档整体表现合规，基本面稳健。"

        summary = (
            f"本报告共识别出 {total} 处风险相关实体，综合风险评定为 **{level}** (得分: {score})。"
        )
        if level in ["高风险", "极高风险"]:
            summary += " 文档中存在显著的潜在负面因素，可能对主体信用或财务稳定性产生重大影响。"
        return summary

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成 Markdown 格式的风险报告 (v2.3 银行级)"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        exec_summary = self._generate_executive_summary(risk)

        report = f"""# 财务风险分析报告: {doc_info['name']}
---
**版本**: v2.3 (Bank Grade) | **生成时间**: {doc_info['analyzed_at']}

## 1. 核心结论 (Executive Summary)
{exec_summary}

## 2. 基础元数据
- **文档分类**: `{classification.get('type', '未知')}`
- **分类置信度**: {classification.get('confidence', 0.0):.4f}
- **分类依据**: {classification.get('reason', '无')}

## 3. 风险量化指标
| 指标 | 评定值 |
| :--- | :--- |
| **风险等级** | **{risk['risk_level']}** |
| **量化得分** | {risk['total_risk_score']} |
| **检测实体数** | {risk['total_entities']} |

## 4. 详细风险实体清单
| 类型 | 实体文本 | 分数 | 置信度 | 识别来源 |
| :--- | :--- | :---: | :---: | :---: |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.4f} | {entity['source']} |\n"
            )

        report += "\n## 5. 风险控制与处置建议 (Mitigation Suggestions)\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "1. 🛑 **深度审计**: 建议启动专项审计流程，重点复核高分风险项。\n"
                "2. 🔍 **加强尽调**: 需对相关关联方进行多维度的背景调查。\n"
                "3. ⏳ **风险计提**: 考虑在财务层面进行适当的风险拨备。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "1. ⚠️ **持续监测**: 建议将该主体列入季度重点观察名单。\n"
                "2. 📝 **补充说明**: 要求对方对报告中识别出的风险点提供书面解释。\n"
            )
        else:
            report += (
                "1. ✅ **标准操作**: 当前风险处于可控范围，建议按标准流程归档，定期复审即可。\n"
            )

        report += f"\n---\n*Disclaimer: 本报告由 Finance-Risk-RAG v2.3 自动生成，仅供内部参考。*"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info("Report saved to %s", output_path)
            save_json_file(analysis_data, output_path.with_suffix(".json"))

        return report

    def process_batch(self, directory: Path) -> List[Dict[str, Any]]:
        """批量处理目录下的所有 PDF"""
        results = []
        for pdf in directory.glob("*.pdf"):
            try:
                results.append(self.analyze_document(pdf))
            except Exception as exc:
                logger.error("Failed to analyze %s: %s", pdf.name, exc)
        return results

    def query_risk(self, question: str):
        """执行 RAG 风险问答"""
        return self.engine.query(question)
