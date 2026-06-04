"""
Finance-Risk-RAG 统一命令行入口
==============================

支持文档处理、实体提取和 RAG 查询。
"""

import argparse
import sys
from pathlib import Path

from src.finance_risk_rag.config import get_config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.processor import DocumentProcessor


def cmd_process(args):
    """处理文档命令"""
    print(f"[*] 开始批量处理文档: {args.input or '默认目录'}")
    processor = DocumentProcessor()
    input_dir = Path(args.input) if args.input else None
    stats = processor.process_batch(input_dir)
    print(f"[+] 处理完成: {stats}")


def cmd_extract(args):
    """提取实体命令"""
    print(f"[*] 开始提取风险实体: {args.input}")
    pipeline = EntityExtractionPipeline()
    pipeline.initialize()

    input_path = Path(args.input)
    result = pipeline.process(input_path)

    output_path = Path(args.output)
    pipeline.save_result(result, output_path)

    print(f"[+] 提取完成！")
    print(f"    实体数: {len(result.entities)}")
    print(f"    总风险: {result.total_risk_score} ({result.risk_level})")

    if not args.no_qa:
        pipeline.interactive_qa(result.entities)


def cmd_query(args):
    """查询命令"""
    engine = RAGEngine()

    if args.build:
        print("[*] 正在构建向量索引...")
        stats = engine.build_index()
        print(f"[+] 索引构建完成: {stats}")

    if args.ask:
        print(f"[*] 查询: {args.ask}")
        result = engine.query(args.ask, top_k=args.top_k)
        print(f"\n回答: {result.answer}")
        if result.sources:
            print("\n来源:")
            for i, src in enumerate(result.sources, 1):
                print(f"  {i}. {src.get('source')} (页码: {src.get('chunk_index')})")


def main():
    parser = argparse.ArgumentParser(
        description="Finance-Risk-RAG: 银行级财务文本风控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # process 子命令
    parser_proc = subparsers.add_parser("process", help="OCR 识别与文档分类")
    parser_proc.add_argument("--input", type=str, help="输入 PDF 目录")
    parser_proc.set_defaults(func=cmd_process)

    # extract 子命令
    parser_ext = subparsers.add_parser("extract", help="风险实体提取")
    parser_ext.add_argument("--input", type=str, default="docs/all_extracted.txt", help="输入文本路径")
    parser_ext.add_argument("--output", type=str, default="docs/entities_extracted.json", help="输出 JSON 路径")
    parser_ext.add_argument("--no-qa", action="store_true", help="禁用交互式问答")
    parser_ext.set_defaults(func=cmd_extract)

    # query 子命令
    parser_query = subparsers.add_parser("query", help="RAG 智能问答")
    parser_query.add_argument("--build", action="store_true", help="构建/更新向量索引")
    parser_query.add_argument("--ask", type=str, help="输入问题")
    parser_query.add_argument("--top-k", type=int, default=4, help="检索相关片段数")
    parser_query.set_defaults(func=cmd_query)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 执行对应函数
    args.func(args)


if __name__ == "__main__":
    main()
