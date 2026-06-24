"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层，协调文档处理、实体提取和 RAG 引擎。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
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
        """延迟初始化文档处理器"""
        if self._processor is None:
            self._processor = DocumentProcessor(self.config)
        return self._processor

    @property
    def pipeline(self) -> EntityExtractionPipeline:
        """延迟初始化实体提取管道"""
        if self._pipeline is None:
            self._pipeline = EntityExtractionPipeline(self.config)
        return self._pipeline

    @property
    def engine(self) -> RAGEngine:
        """延迟初始化 RAG 引擎"""
        if self._engine is None:
            self._engine = RAGEngine(self.config)
        return self._engine

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API"""
        return self.pipeline

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """执行单文档完整分析：OCR -> 分类 -> 实体提取。

        Args:
            pdf_path: PDF 文件的路径。

        Returns:
            包含文档信息、分类结果和风险分析的字典。
        """
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

    def run_full_analysis(self, input_path: Path) -> Dict[str, Any]:
        """执行全流程分析：OCR -> 文档分类 -> 实体提取 -> RAG 索引构建。

        始终返回包含状态、结果和计数的标准字典。

        Args:
            input_path: PDF 文件或包含 PDF 的目录路径。

        Returns:
            标准化结果字典，例如：{"status": "success", "results": [...], "count": 1}。
        """
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            analysis_data = self.analyze_document(input_path)

            txt_path = input_path.with_suffix(".txt")
            if txt_path.exists():
                self.engine.add_documents([txt_path])

            return {"status": "success", "results": [analysis_data], "count": 1}

        results_list = []
        if input_path.is_dir():
            self.processor.process_directory(input_path)
            txt_files = list(input_path.glob("*.txt"))
            for txt_file in txt_files:
                if txt_file.name == "all_extracted.txt":
                    continue

                # 为了保持 run_full_analysis 的一致性，我们在这里重新组装分析数据
                # 或者调用 analyze_document（如果 PDF 存在）
                pdf_path = txt_file.with_suffix(".pdf")
                if pdf_path.exists():
                    try:
                        results_list.append(self.analyze_document(pdf_path))
                    except Exception as e:
                        logger.error(f"Error analyzing {pdf_path}: {e}")
                else:
                    # 如果只有 txt，则仅做提取
                    extraction_res = self.pipeline.process(txt_file)
                    results_list.append(
                        {
                            "document_info": {"name": txt_file.name, "path": str(txt_file)},
                            "risk_analysis": extraction_res.to_dict(),
                        }
                    )

            self.engine.build_index()

        return {"status": "success", "results": results_list, "count": len(results_list)}

    def generate_report(
        self, analysis_data: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """根据分析数据生成 Markdown 格式的风险报告"""
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
