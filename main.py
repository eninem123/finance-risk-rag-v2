"""
Finance-Risk-RAG 统一命令行入口
"""

import argparse
import sys
from pathlib import Path

# 将 src 目录添加到 Python 路径
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag.config import get_config
from finance_risk_rag.extract_text import DocumentProcessor
from finance_risk_rag.extract_entities import EntityExtractionPipeline
from finance_risk_rag.rag_core import RAGEngine


def main():
    parser = argparse.ArgumentParser(
        description="Finance-Risk-RAG: 银行级多语言财务文本风控AI系统"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # process 子命令
    process_parser = subparsers.add_parser("process", help="PDF 文档 OCR 识别与分类")
    process_parser.add_argument("--input", type=str, default="docs", help="输入 PDF 目录")

    # extract 子命令
    extract_parser = subparsers.add_parser("extract", help="风险实体抽取")
    extract_parser.add_argument("--input", type=str, default="docs/all_extracted.txt", help="输入文本文件")
    extract_parser.add_argument("--output", type=str, default="docs/entities_extracted.json", help="输出 JSON 文件")
    extract_parser.add_argument("--no-qa", action="store_true", help="禁用交互式问答")

    # query 子命令
    query_parser = subparsers.add_parser("query", help="RAG 智能问答")
    query_parser.add_argument("--build-db", action="store_true", help="构建向量数据库")
    query_parser.add_argument("--q", type=str, help="执行查询问题")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索分块数量")

    args = parser.parse_args()

    if args.command == "process":
        processor = DocumentProcessor()
        processor.batch_process(args.input)

    elif args.command == "extract":
        pipeline = EntityExtractionPipeline()
        pipeline.initialize()
        result = pipeline.process(Path(args.input))
        pipeline.save_result(result, Path(args.output))

        print(f"\n实体提取完成！")
        print(f"  实体数: {len(result.entities)}")
        print(f"  总风险: {result.total_risk_score} ({result.risk_level})")

        if not args.no_qa:
            pipeline.interactive_qa(result.entities)

    elif args.command == "query":
        engine = RAGEngine()
        if args.build_db:
            stats = engine.build_index()
            print(f"索引构建完成: {stats}")

        if args.q:
            result = engine.query(args.q, top_k=args.top_k)
            print(f"\n回答: {result.answer}")
            print("\n来源:")
            for i, source in enumerate(result.sources, 1):
                print(f"  {i}. {source.get('source')} (分块 {source.get('chunk_index')})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
