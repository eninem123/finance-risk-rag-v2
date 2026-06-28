"""
Finance-Risk-RAG 业务分析服务层
==============================

业务编排层，协调文档处理、实体提取和 RAG 引擎。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Config, get_config
from .engine import RAGEngine
from .exceptions import FinanceRiskRAGError
from .extractor import EntityExtractionPipeline
from .llm import LLMClientWrapper
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
        self.processor = processor or DocumentProcessor(self.config)
        self.pipeline = pipeline or extractor or EntityExtractionPipeline(self.config)
        self.engine = engine or RAGEngine(self.config)
        self.llm_client = LLMClientWrapper()

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API"""
        return self.pipeline

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """执行单文档完整分析：OCR -> 分类 -> 实体提取"""
        logger.info("Starting full analysis for %s", pdf_path.name)

        try:
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
        except FinanceRiskRAGError as e:
            logger.error(f"Analysis failed for {pdf_path.name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error analyzing {pdf_path.name}: {e}")
            raise FinanceRiskRAGError(f"Unexpected error: {e}")

    def run_full_analysis(
        self, input_path: Path
    ) -> Union[Dict[str, Any], Dict[str, ExtractionResult], None]:
        """
        执行全流程分析：OCR -> 文档分类 -> 实体提取 -> RAG 索引构建。
        """
        try:
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
        except Exception as e:
            logger.error(f"Failed to analyze {input_path}: {e}")
            return None

    def _generate_executive_summary(self, risk_data: Dict[str, Any]) -> str:
        """生成 AI 摘要"""
        if not self.llm_client.is_available:
            return "（摘要由于 AI 服务不可用而无法生成）"

        entities_summary = ", ".join([f"{e['text']}({e['type']})" for e in risk_data["entities"][:5]])
        prompt = f"""
请为以下财务风险分析结果生成一段专业的行政摘要。
风险等级：{risk_data['risk_level']}
风险分数：{risk_data['total_risk_score']}
核心实体：{entities_summary}

要求：
1. 语言简练专业（银行级风格）
2. 重点突出主要风险点
3. 字数在 150 字以内
"""
        try:
            return self.llm_client.chat([{"role": "user", "content": prompt}])
        except Exception:
            return "（摘要生成失败）"

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成 Markdown 格式的风险报告"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        summary = self._generate_executive_summary(risk)

        report = f"""# 财务风险分析报告: {doc_info['name']}

## 1. 核心摘要
{summary}

## 2. 基本信息
- **分析时间**: {doc_info['analyzed_at']}
- **文档类型**: {classification.get('type', '未知')} (置信度: {classification.get('confidence', 0.0):.2f})
- **分类依据**: {classification.get('reason', '无')}

## 3. 风险评估
- **综合风险等级**: **{risk['risk_level']}**
- **风险量化评分**: {risk['total_risk_score']}
- **识别风险实体数**: {risk['total_entities']}

## 4. 详细风险清单
| 类型 | 实体文本 | 风险分数 | 置信度 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.2f} | {entity['source']} |\n"
            )

        report += "\n## 5. 结论与审计建议\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "### 🛑 审计红线提示\n"
                "- **加强尽职调查**: 建议启动穿透式审计，核实相关关联交易真实性。\n"
                "- **限制信用额度**: 鉴于检测到的多项高风险实体，建议暂缓信贷审批。\n"
                "- **现场核查**: 必须派人实地走访，核实财报中关键数据的真实性。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "### ⚠️ 合规风险提示\n"
                "- **常规监控**: 建议将该实体列入月度风险观察名单。\n"
                "- **补充资料**: 要求客户提供涉及风险实体的详细交易明细和支撑材料。\n"
                "- **财务访谈**: 组织财务总监级别的专项面谈。\n"
            )
        else:
            report += (
                "### ✅ 流程建议\n"
                "- **正常推进**: 风险可控，建议按标准流程执行。\n"
                "- **动态跟踪**: 每季度进行一次自动化的回溯扫描即可。\n"
            )

        report += f"\n---\n*报告由 Finance-Risk-RAG v2.3 自动生成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"

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
