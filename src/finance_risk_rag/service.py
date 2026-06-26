"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层，协调文档处理、实体提取和 RAG 引擎。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError

from .config import Config, get_config
from .engine import RAGEngine
from .exceptions import ExtractionError, FinanceRiskRAGError, LLMError, OCRError
from .extractor import EntityExtractionPipeline
from .models import ExtractionResult
from .processor import DocumentProcessor
from .utils import save_json_file

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """风险分析服务编排类，提供高级业务接口。"""

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
        except (OCRError, ExtractionError) as e:
            logger.error(f"Critical error during analysis of {pdf_path.name}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error analyzing {pdf_path.name}: {e}")
            raise FinanceRiskRAGError(f"Analysis failed: {e}") from e

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
                try:
                    extraction_res = self.pipeline.process(txt_file)
                    results[txt_file.stem + ".pdf"] = extraction_res
                except Exception as e:
                    logger.error(f"Failed to extract entities from {txt_file.name}: {e}")

            self.engine.build_index()

        return results

    def _generate_executive_summary(self, risk_level: str, entities: List[Dict[str, Any]]) -> str:
        """生成执行摘要 (Executive Summary)"""
        if not self.engine.llm_client.is_available:
            return "执行摘要生成失败：LLM 客户端不可用。"

        prompt = f"""
请作为资深风控官，根据以下风险数据生成一段 150 字以内的专业执行摘要。

【风险数据】
- 风险等级：{risk_level}
- 关键风险点：{", ".join([e['text'] for e in entities[:5]])}

【要求】
- 语气专业、严谨。
- 重点指出最显着的风险因素。
- 给出明确的风险态度（通过/关注/预警）。
"""
        try:
            return self.engine.llm_client.chat([{"role": "user", "content": prompt}], max_tokens=300)
        except Exception:
            return "无法自动生成摘要，请参考下方详细风险列表。"

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成专业 Markdown 格式的风险报告"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]

        summary = self._generate_executive_summary(risk["risk_level"], risk["entities"])

        report = f"""# 🏦 财务风险分析专业报告: {doc_info['name']}

> **报告版本**: v2.3 (Enterprise)
> **分析时间**: {doc_info['analyzed_at']}
> **机密等级**: 内部参阅 (Internal Only)

---

## 📋 1. 执行摘要 (Executive Summary)
{summary}

---

## 🔍 2. 基础信息与分类
- **文档名称**: `{doc_info['name']}`
- **识别类别**: **{classification.get('type', '未知')}**
- **分类置信度**: `{classification.get('confidence', 0.0):.2%}`
- **分类依据**: {classification.get('reason', '无')}

---

## ⚖️ 3. 风险评估摘要
- **综合风险等级**: **{risk['risk_level']}**
- **量化风险评分**: `{risk['total_risk_score']}`
- **识别实体总数**: `{risk['total_entities']}`

---

## 🚩 4. 详细风险实体清单
| 风险类型 | 实体文本 | 风险权重 | 置信度 | 来源渠道 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | `{entity['text']}` | {entity['risk_score']} | "
                f"{entity['confidence']:.2%} | {entity['source']} |\n"
            )

        report += "\n---\n\n## 💡 5. 专家建议 (Mitigation Suggestions)\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "### 🔴 预警：建议立即启动深度尽调\n"
                "1. **人工审计**: 建议指派高级审计师对上述风险点进行核查。\n"
                "2. **现场尽调**: 前往相关实体进行现场走访和财务真实性核验。\n"
                "3. **风险隔离**: 在核验完成前，建议暂停相关信贷或业务流程。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "### 🟡 关注：建议进行合规性补充说明\n"
                "1. **资料补充**: 要求对方提供针对识别出的风险点的补充说明材料。\n"
                "2. **持续监控**: 将该实体纳入月度风险监控名单。\n"
            )
        else:
            report += (
                "### 🟢 通过：常规业务处理\n"
                "1. **正常推进**: 暂未发现重大负面风险，可按标准流程处理。\n"
                "2. **归档备案**: 将本次分析结果存入客户风控档案。\n"
            )

        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(report, encoding="utf-8")
                logger.info("Report saved to %s", output_path)
                save_json_file(analysis_data, output_path.with_suffix(".json"))
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {e}")

        return report

    def process_batch(self, directory: Path) -> List[Dict[str, Any]]:
        """批量处理目录下的所有 PDF"""
        results = []
        for pdf in directory.glob("*.pdf"):
            try:
                results.append(self.analyze_document(pdf))
            except FinanceRiskRAGError as exc:
                logger.error("Domain error analyzing %s: %s", pdf.name, exc)
            except Exception as exc:
                logger.error("Unexpected error analyzing %s: %s", pdf.name, exc)
        return results

    def query_risk(self, question: str):
        """执行 RAG 风险问答"""
        try:
            return self.engine.query(question)
        except (LLMError, FinanceRiskRAGError) as e:
            logger.error(f"RAG Query failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected failure during query: {e}")
            raise FinanceRiskRAGError(f"Query processing error: {e}") from e
