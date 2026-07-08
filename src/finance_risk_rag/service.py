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

    def _generate_executive_summary(self, analysis_data: Dict[str, Any]) -> str:
        """使用 LLM 生成分析摘要"""
        if not self.processor.llm_client.is_available:
            return "（AI 摘要功能暂不可用）"

        risk = analysis_data["risk_analysis"]
        entities_summary = ", ".join([e["text"] for e in risk["entities"][:5]])

        prompt = f"""
请根据以下风险分析结果，撰写一段专业的银行级风险分析执行摘要（150字以内）。

【文档信息】
- 文档名称: {analysis_data['document_info']['name']}
- 文档类别: {analysis_data['classification']['type']}
- 风险等级: {risk['risk_level']} (得分: {risk['total_risk_score']})

【关键风险实体】
{entities_summary}

【摘要要求】
- 语气客观专业
- 重点阐述该文档对银行信贷或投资决策的潜在影响
"""
        try:
            return self.processor.llm_client.chat([{"role": "user", "content": prompt}])
        except Exception:
            return "（摘要生成失败）"

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成 Markdown 格式的专业风险报告"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        summary = self._generate_executive_summary(analysis_data)

        report = f"""# 财务风险分析报告: {doc_info['name']}

## 1. 业务执行摘要 (Executive Summary)
{summary}

## 2. 基础档案
- **分析流水号**: `REF-{doc_info['hash'][:8].upper()}`
- **分析时间**: {doc_info['analyzed_at']}
- **文档类型识别**: {classification.get('type', '未知')}
- **分类置信度**: {classification.get('confidence', 0.0):.2%}
- **分类依据**: {classification.get('reason', '无')}

## 3. 风险量化评估
| 评估维度 | 指标值 |
| :--- | :--- |
| **风险等级** | **{risk['risk_level']}** |
| **综合风险评分** | {risk['total_risk_score']} |
| **识别风险实体数** | {risk['total_entities']} |

## 4. 关键风险实体清单
| 风险类别 | 实体内容 | 风险评分 | 置信度 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.2f} | {entity['source']} |\n"
            )

        report += "\n## 5. 专家结论与建议\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "🚨 **专家结论**: 该文档包含多项高危财务风险因素，违约风险较高。\n\n"
                "🛡️ **风控建议**:\n"
                "1. **暂停流程**: 建议立即暂停相关信贷或投资审批流程。\n"
                "2. **深度穿透**: 需对识别出的风险实体进行穿透式背景调查。\n"
                "3. **现场核查**: 派遣专员进行现场实物资产或账簿核查。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "⚠️ **专家结论**: 存在一定比例的风险瑕疵，需保持警惕。\n\n"
                "🛡️ **风控建议**:\n"
                "1. **关注名单**: 将客户列入重点观察名单。\n"
                "2. **补充说明**: 要求客户对识别出的特定风险点提交书面解释资料。\n"
            )
        else:
            report += (
                "✅ **专家结论**: 暂未发现重大系统性风险，处于可控范围。\n\n"
                "🛡️ **风控建议**: 按标准业务流程进行后续操作，保持定期监测即可。\n"
            )

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
