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
        """
        初始化风险分析服务。

        采用延迟加载（Lazy Initialization）模式，避免在不需要时初始化重型组件。
        """
        self.config = config or get_config()
        self._processor = processor
        self._pipeline = pipeline or extractor
        self._engine = engine

    @property
    def processor(self) -> DocumentProcessor:
        """获取文档处理器"""
        if self._processor is None:
            self._processor = DocumentProcessor(self.config)
        return self._processor

    @property
    def pipeline(self) -> EntityExtractionPipeline:
        """获取实体提取管道"""
        if self._pipeline is None:
            self._pipeline = EntityExtractionPipeline(self.config)
        return self._pipeline

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API，等同于 pipeline"""
        return self.pipeline

    @property
    def engine(self) -> RAGEngine:
        """获取 RAG 引擎"""
        if self._engine is None:
            self._engine = RAGEngine(self.config)
        return self._engine

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
            logger.info("Starting batch processing for directory: %s", input_path)
            self.processor.process_directory(input_path)
            txt_files = list(input_path.glob("*.txt"))
            for txt_file in txt_files:
                if txt_file.name == "all_extracted.txt":
                    continue
                extraction_res = self.pipeline.process(txt_file)
                results[txt_file.stem + ".pdf"] = extraction_res

            self.engine.build_index()
            logger.info("Batch processing complete. Analyzed %d documents.", len(results))

        return results

    def build_index(self):
        """构建 RAG 索引"""
        self.engine.build_index()

    def generate_batch_reports(self, input_dir: Path, output_dir: Path) -> List[str]:
        """批量生成风险报告"""
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_results = self.process_batch(input_dir)
        report_paths = []
        for analysis in batch_results:
            name = analysis["document_info"]["name"]
            report_path = output_dir / f"{Path(name).stem}_report.md"
            self.generate_report(analysis, report_path)
            report_paths.append(str(report_path))
        return report_paths

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
