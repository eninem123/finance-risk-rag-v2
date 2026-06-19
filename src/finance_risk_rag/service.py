"""
Finance-Risk-RAG 风险分析服务层
==============================

业务编排层，协调文档处理、实体提取和 RAG 引擎。
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from .config import Config, get_config
from .engine import RAGEngine
from .extractor import EntityExtractionPipeline
from .models import ExtractionResult
from .processor import DocumentProcessor

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
        self.config = config or get_config()
        self.processor = processor or DocumentProcessor(self.config)
        self.pipeline = pipeline or EntityExtractionPipeline(self.config)
        self.engine = engine or RAGEngine(self.config)

    def run_full_analysis(self, input_path: Path) -> Dict[str, ExtractionResult]:
        """
        执行全流程分析：OCR -> 文档分类 -> 实体提取 -> RAG 索引构建
        """
        results: Dict[str, ExtractionResult] = {}

        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            # 处理单个文件
            proc_res = self.processor.process_single_pdf(input_path)
            extracted_text = proc_res["text"]
            extraction_res = self.pipeline.process(extracted_text)
            results[input_path.name] = extraction_res

            # 更新 RAG 索引
            txt_path = input_path.with_suffix(".txt")
            if txt_path.exists():
                self.engine.add_documents([txt_path])

        elif input_path.is_dir():
            # 处理目录
            self.processor.process_directory(input_path)
            # 对目录下的每个文本文件执行提取
            txt_files = list(input_path.glob("*.txt"))
            for txt_file in txt_files:
                if txt_file.name == "all_extracted.txt":
                    continue
                extraction_res = self.pipeline.process(txt_file)
                results[txt_file.stem + ".pdf"] = extraction_res

            # 构建 RAG 索引
            self.engine.build_index()

        return results

    def query_risk(self, question: str):
        """执行 RAG 风险问答"""
        return self.engine.query(question)
