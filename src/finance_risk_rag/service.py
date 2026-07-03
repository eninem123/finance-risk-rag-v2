"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层，采用延迟初始化模式，协调文档处理、实体提取和 RAG 引擎。
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
    """
    风险分析服务编排类。
    采用懒加载模式初始化各个组件，以优化资源使用。
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        processor: Optional[DocumentProcessor] = None,
        pipeline: Optional[EntityExtractionPipeline] = None,
        engine: Optional[RAGEngine] = None,
    ):
        self.config = config or get_config()
        self._processor = processor
        self._pipeline = pipeline
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
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API 访问"""
        return self.pipeline

    @property
    def engine(self) -> RAGEngine:
        if self._engine is None:
            self._engine = RAGEngine(self.config)
        return self._engine

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """执行单文档完整分析流程：OCR -> 分类 -> 实体提取"""
        logger.info(f"Starting full analysis for {pdf_path.name}")

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
        运行全流程分析并构建 RAG 索引。
        """
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            analysis_data = self.analyze_document(input_path)

            # 为 RAG 添加文档
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

            # 构建/更新索引
            self.engine.build_index()

        return results

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """
        生成专业级 Markdown 财务风险报告。
        """
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        report = f"""# 财务风险分析报告: {doc_info['name']}

## 1. 报告摘要 (Executive Summary)
- **分析时间**: `{doc_info['analyzed_at']}`
- **文档分类**: **{classification.get('type', '未知')}** (置信度: {classification.get('confidence', 0.0):.2%})
- **分类依据**: {classification.get('reason', 'N/A')}
- **最终风险评级**: **{risk['risk_level']}**
- **量化风险总分**: `{risk['total_risk_score']}`

---

## 2. 风险实体分布
| 实体类别 | 实体内容 | 风险分值 | 置信度 | 来源渠道 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.2%} | {entity['source']} |\n"
            )

        report += """
---

## 3. 风险控制与处置建议
"""
        level = risk["risk_level"]
        if level in ["极高风险", "高风险"]:
            report += (
                "🚨 **高危警示**: 该文档涉及多个敏感风险点，建议立即启动专项调查，"
                "并暂停相关业务往来。建议核实相关实体的资信背景及涉诉情况。\n"
            )
        elif level == "中风险":
            report += (
                "⚠️ **中度关注**: 存在一定的财务合规或信用风险。建议进一步核查相关交易细节，"
                "并要求补充支撑性材料，必要时进行电话回访。\n"
            )
        else:
            report += (
                "✅ **合规建议**: 风险指标在正常范围内。建议按标准流程执行，"
                "并将其纳入常规监控名单。\n"
            )

        report += f"\n---\n*Report generated by Finance-Risk-RAG v2.3 (Automated System)*"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info(f"Report saved to {output_path}")
            save_json_file(analysis_data, output_path.with_suffix(".json"))

        return report

    def process_batch(self, directory: Path) -> List[Dict[str, Any]]:
        """批量处理文档"""
        results = []
        for pdf in directory.glob("*.pdf"):
            try:
                results.append(self.analyze_document(pdf))
            except Exception as exc:
                logger.error(f"Failed to analyze {pdf.name}: {exc}")
        return results

    def query_risk(self, question: str):
        """执行 RAG 风险问答"""
        return self.engine.query(question)
