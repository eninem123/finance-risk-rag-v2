"""
Finance-Risk-RAG 统一命令行入口
==============================

支持文档处理、实体提取、风险问答等功能。
"""

import argparse
import sys
from pathlib import Path

# 将 src 目录添加到路径中
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag.config import get_config
from finance_risk_rag.extract_entities import EntityExtractionPipeline
from finance_risk_rag.extract_text import DocumentProcessor
from finance_risk_rag.rag_core import RAGEngine


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 银行级财务文本风控系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 1. 文档处理 (OCR + 分类)
    parser_process = subparsers.add_parser("process", help="批量处理 PDF 文档 (OCR + 分类)")

    # 2. 实体提取
    parser_extract = subparsers.add_parser("extract", help="从提取的文本中识别风险实体")
    parser_extract.add_argument("--input", type=str, default="docs/all_extracted.txt", help="输入文本路径")
    parser_extract.add_argument("--output", type=str, default="docs/entities_extracted.json", help="结果保存路径")

    # 3. RAG 问答
    parser_query = subparsers.add_parser("query", help="基于文档内容执行 RAG 风险问答")
    parser_query.add_argument("question", type=str, help="您的问题")
    parser_query.add_argument("--build", action="store_true", help="是否先重建索引")
    parser_query.add_argument("--top-k", type=int, default=4, help="检索相关片段数量")

    args = parser.parse_args()

    if args.command == "process":
        print(">>> 启动批量文档处理...")
        processor = DocumentProcessor()
        processor.batch_process()
        print(">>> 处理完成。")

    elif args.command == "extract":
        print(f">>> 启动实体提取: {args.input}")
        pipeline = EntityExtractionPipeline()
        pipeline.initialize()
        result = pipeline.process(Path(args.input))
        pipeline.save_result(result, Path(args.output))

        print(f"\n提取完成！")
        print(f"  实体数: {len(result.entities)}")
        print(f"  风险等级: {result.risk_level} (总分: {result.total_risk_score})")
        print(f"  结果已保存至: {args.output}")

    elif args.command == "query":
        engine = RAGEngine()
        if args.build:
            print(">>> 正在构建向量索引...")
            engine.build_index()

        print(f">>> 执行查询: {args.question}")
        result = engine.query(args.question, top_k=args.top_k)

        print(f"\n回答:")
        print(f"{result.answer}")
        print(f"\n来源:")
        for source in result.sources:
            print(f"  - {source.get('source')} (分块 {source.get('chunk_index')})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
