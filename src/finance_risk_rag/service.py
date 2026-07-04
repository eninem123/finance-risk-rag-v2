"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层 (v2.3)，提供延迟初始化和标准化的 API。
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
    ):
        self._config = config
        self._processor = processor
        self._pipeline = pipeline
        self._engine = engine

    @property
    def config(self) -> Config:
        if self._config is None:
            self._config = get_config()
        return self._config

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
            "status": "success",
            "document_info": {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "hash": proc_result["hash"],
                "analyzed_at": datetime.now().isoformat(),
            },
            "classification": proc_result["classification"],
            "risk_analysis": extraction_result.to_dict(),
        }

    def run_full_analysis(self, input_path: Path) -> Dict[str, Any]:
        """
        执行全流程分析 (v2.3)
        统一返回结构化的字典。
        """
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            analysis_data = self.analyze_document(input_path)

            txt_path = input_path.with_suffix(".txt")
            if txt_path.exists():
                self.engine.add_documents([txt_path])

            return {"status": "success", "results": analysis_data, "count": 1}

        if input_path.is_dir():
            self.processor.process_directory(input_path)
            txt_files = list(input_path.glob("*.txt"))

            results = {}
            for txt_file in txt_files:
                if txt_file.name == "all_extracted.txt":
                    continue
                extraction_res = self.pipeline.process(txt_file)
                results[txt_file.stem + ".pdf"] = extraction_res.to_dict()

            self.engine.build_index()
            return {"status": "success", "results": results, "count": len(results)}

        return {"status": "error", "message": "Invalid input path"}

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成银行级 Markdown 格式的风险报告 (v2.3)"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        report = f"""# 🏦 财务风险分析报告 (v2.3)

## 1. 执行摘要 (Executive Summary)
- **文档名称**: `{doc_info['name']}`
- **分析时间**: {doc_info['analyzed_at']}
- **文档分类**: **{classification.get('type', '未知')}** (置信度: {classification.get('confidence', 0.0):.4f})
- **综合风险等级**: <span style="color:{self._get_risk_color(risk['risk_level'])}">**{risk['risk_level']}**</span>
- **量化风险评分**: `{risk['total_risk_score']}`

## 2. 风险概览
系统在文档中识别出 **{risk['total_entities']}** 个关键风险实体点。以下为分项统计：

| 风险维度 | 实体数量 | 最高风险评分 |
| :--- | :---: | :---: |
| 规则匹配 | {len([e for e in risk['entities'] if e['source'] == 'rule'])} | {max([e['risk_score'] for e in risk['entities'] if e['source'] == 'rule'] + [0])} |
| AI 模型识别 | {len([e for e in risk['entities'] if e['source'] == 'bert'])} | {max([e['risk_score'] for e in risk['entities'] if e['source'] == 'bert'] + [0])} |

## 3. 详细风险清单
下表列出了识别出的所有风险实体及其详细信息：

| 实体类型 | 文本内容 | 风险评分 | 置信度 | 来源 |
| :--- | :--- | :---: | :---: | :---: |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | `{entity['text']}` | {entity['risk_score']} | "
                f"{entity['confidence']:.4f} | {entity['source']} |\n"
            )

        report += "\n## 4. 专家建议与控制措施\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "🔴 **关键预警**: 该文档存在严重风险隐患，建议立即启动二级审查流程。\n"
                "- 建议加强对相关关联方的尽职调查。\n"
                "- 对提及的资金异常点进行专项核查。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "🟡 **关注提示**: 存在中等程度的风险点，建议进行常规性的背景穿透分析。\n"
                "- 关注后续相关信息的补充披露。\n"
            )
        else:
            report += "🟢 **正常合规**: 未发现显著异常，建议按常规流程备案。\n"

        report += "\n---\n*报告由 Finance-Risk-RAG v2.3 自动生成。*"

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info("Report saved to %s", output_path)
            save_json_file(analysis_data, output_path.with_suffix(".json"))

        return report

    def _get_risk_color(self, level: str) -> str:
        colors = {"低风险": "green", "中风险": "orange", "高风险": "red", "极高风险": "darkred"}
        return colors.get(level, "gray")

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
