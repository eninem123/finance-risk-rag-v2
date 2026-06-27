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
        llm_client: Optional[LLMClientWrapper] = None,
    ):
        self.config = config or get_config()
        self.llm_client = llm_client or LLMClientWrapper(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
        )
        self.processor = processor or DocumentProcessor(self.config, llm_client=self.llm_client)
        self.pipeline = pipeline or EntityExtractionPipeline(self.config)
        self.engine = engine or RAGEngine(self.config, llm_client=self.llm_client)

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API"""
        return self.pipeline

    def _generate_executive_summary(self, analysis_data: Dict[str, Any]) -> str:
        """使用 LLM 生成执行摘要"""
        if not self.llm_client.is_available:
            return "（由于未配置 LLM API，无法生成 AI 执行摘要）"

        risk = analysis_data["risk_analysis"]
        classification = analysis_data["classification"]
        entities_summary = ", ".join([f"{e['text']}({e['type']})" for e in risk["entities"][:10]])

        prompt = [
            {
                "role": "system",
                "content": "你是一名银行风控官，负责根据自动化提取的风险实体编写简短的分析摘要。",
            },
            {
                "role": "user",
                "content": f"""
请根据以下数据生成一段 150 字以内的风险执行摘要：

文档类型：{classification.get('type')}
风险等级：{risk['risk_level']}
风险总分：{risk['total_risk_score']}
识别到的核心实体：{entities_summary}

摘要应包含对文档性质的判断及最关键的风险警示。
""",
            },
        ]
        try:
            return self.llm_client.chat(prompt, max_tokens=300)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return "生成执行摘要时出错。"

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """执行单文档完整分析：OCR -> 分类 -> 实体提取"""
        logger.info("Starting full analysis for %s", pdf_path.name)

        proc_result = self.processor.process_single_pdf(pdf_path)
        extraction_result = self.pipeline.process(proc_result["text"])

        analysis_data = {
            "document_info": {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "hash": proc_result["hash"],
                "analyzed_at": datetime.now().isoformat(),
            },
            "classification": proc_result["classification"],
            "risk_analysis": extraction_result.to_dict(),
        }

        # 添加 AI 摘要
        analysis_data["executive_summary"] = self._generate_executive_summary(analysis_data)
        return analysis_data

    def run_full_analysis(
        self, input_path: Path
    ) -> Union[Dict[str, ExtractionResult], Dict[str, Any]]:
        """
        执行全流程分析：OCR -> 文档分类 -> 实体提取 -> RAG 索引构建。
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

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成 Markdown 格式的风险报告"""
        doc_info = analysis_data["document_info"]
        classification = analysis_data["classification"]
        risk = analysis_data["risk_analysis"]
        summary = analysis_data.get("executive_summary", "无")

        report = f"""# 财务风险分析报告: {doc_info['name']}

## 1. 结论摘要 (AI 生成)
> {summary}

## 2. 基本信息
- **分析时间**: {doc_info['analyzed_at']}
- **文档类型**: {classification.get('type', '未知')} (置信度: {classification.get('confidence', 0.0):.2f})
- **分类依据**: {classification.get('reason', '无')}

## 3. 风险评估概览
- **综合风险等级**: **{risk['risk_level']}**
- **量化评分**: {risk['total_risk_score']}
- **识别实体总数**: {risk['total_entities']}

## 4. 详细风险实体分布
| 类型 | 实体文本 | 风险分数 | 置信度 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['type']} | {entity['text']} | {entity['risk_score']} | "
                f"{entity['confidence']:.2f} | {entity['source']} |\n"
            )

        report += "\n## 5. 风控建议\n"
        if risk["risk_level"] in ["高风险", "极高风险"]:
            report += (
                "### 🔴 严控类建议\n"
                "- **立即人工介入**: 该文档包含多项高风险因素，需高级风控员核实。\n"
                "- **现场尽调**: 建议启动针对识别出的风险实体的现场尽职调查。\n"
                "- **关联交易核查**: 深度挖掘实体间的资金往来情况。\n"
            )
        elif risk["risk_level"] == "中风险":
            report += (
                "### 🟡 关注类建议\n"
                "- **补充资料**: 要求客户提供关于相关实体的背景说明文件。\n"
                "- **持续监控**: 将识别出的风险点加入季度监控清单。\n"
            )
        else:
            report += "### 🟢 正常类建议\n- **常规审核**: 风险较低，可按正常流程通过，保留分析存档即可。\n"

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
