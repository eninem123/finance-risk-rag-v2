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
        """延迟加载文档处理器"""
        if self._processor is None:
            self._processor = DocumentProcessor(self.config)
        return self._processor

    @property
    def pipeline(self) -> EntityExtractionPipeline:
        """延迟加载提取管道"""
        if self._pipeline is None:
            self._pipeline = EntityExtractionPipeline(self.config)
        return self._pipeline

    @property
    def extractor(self) -> EntityExtractionPipeline:
        """兼容旧版 API"""
        return self.pipeline

    @property
    def engine(self) -> RAGEngine:
        """延迟加载 RAG 引擎"""
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

        # 生成执行摘要
        summary = self._generate_executive_summary(risk)

        report = f"""# 银行级财务风险分析报告 v2.3

## 1. 执行摘要
{summary}

## 2. 基本信息
- **文档名称**: {doc_info['name']}
- **分析时间**: {doc_info['analyzed_at']}
- **文档类型**: {classification.get('type', '未知')} (置信度: {classification.get('confidence', 0.0):.2f})
- **分类依据**: {classification.get('reason', '无')}

## 3. 风险评估矩阵
- **风险评级**: **{risk['risk_level']}**
- **量化总分**: {risk['total_risk_score']}
- **实体统计**: 共识别出 {risk['total_entities']} 个风险点

### 风险类型分布
"""
        # 统计实体类型
        type_counts: Dict[str, int] = {}
        for entity in risk["entities"]:
            t = entity["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        report += "| 风险类型 | 出现频率 |\n| :--- | :--- |\n"
        for t, count in type_counts.items():
            report += f"| {t} | {count} |\n"

        report += """
## 4. 核心风险清单
| 风险实体 | 类别 | 风险权重 | 影响分 | 来源 |
| :--- | :--- | :--- | :--- | :--- |
"""
        for entity in risk["entities"]:
            report += (
                f"| {entity['text']} | {entity['type']} | {entity['risk_score']} | "
                f"{entity.get('impact_score', 0.0):.2f} | {entity['source']} |\n"
            )

        report += "\n## 5. 风控建议与措施\n"
        report += self._get_structured_suggestions(risk)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info("Report saved to %s", output_path)
            save_json_file(analysis_data, output_path.with_suffix(".json"))

        return report

    def _generate_executive_summary(self, risk: Dict[str, Any]) -> str:
        """生成自动化的执行摘要"""
        if risk["total_entities"] == 0:
            return "本报告未在提交的文档中发现显著的已知财务风险点。建议按常规流程处理。"

        top_entities = sorted(risk["entities"], key=lambda x: x.get("impact_score", 0), reverse=True)
        main_risk = top_entities[0]["text"] if top_entities else "多项财务指标"

        return (
            f"本报告通过自动化的多模态风控系统识别出该文档存在 {risk['risk_level']}。 "
            f"主要风险点聚焦于“{main_risk}”等。 综合量化评分为 {risk['total_risk_score']}，"
            f"建议管理层根据本报告第 5 节的建议采取相应风控措施。"
        )

    def _get_structured_suggestions(self, risk: Dict[str, Any]) -> str:
        """获取结构化的风控建议"""
        level = risk["risk_level"]
        if level in ["高风险", "极高风险"]:
            return """
- [ ] **深度尽调**: 立即启动二级财务尽职调查，核实重点科目。
- [ ] **限制授信**: 在风险未排除前，建议冻结或限制该关联主体的授信额度。
- [ ] **现场核查**: 指派审计团队进行现场实地核查和访谈。
- [ ] **高层介入**: 风险等级已达预警线，需呈报风控委员会审议。
"""
        elif level == "中风险":
            return """
- [ ] **持续监控**: 将该实体列入观察名单，按季度监控其公开财务变动。
- [ ] **补充资料**: 要求客户提供关联交易详情或抵押物评估更新。
- [ ] **电话访谈**: 针对特定风险点（如现金流波动）进行专项电话问询。
"""
        else:
            return """
- [x] **常规监测**: 纳入年度常规复核计划。
- [x] **合规备案**: 分析结果已存入信贷管理系统备案。
"""

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
