"""
Finance-Risk-RAG 统一命令行界面
"""

import argparse
import sys
from pathlib import Path

# 将 src 添加到 sys.path 以确保可以导入 finance_risk_rag
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from finance_risk_rag.extract_entities import EntityExtractionPipeline  # noqa: E402
from finance_risk_rag.extract_text import DocumentProcessor  # noqa: E402
from finance_risk_rag.rag_core import RAGEngine  # noqa: E402


def cmd_process(args):
    """处理 OCR 和分类"""
    processor = DocumentProcessor()
    processor.batch_process()
    print("文档处理完成。")


def cmd_extract(args):
    """提取风险实体"""
    pipeline = EntityExtractionPipeline()
    pipeline.initialize()
    result = pipeline.process(Path(args.input))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"实体提取完成。结果保存至: {args.output}")
    print(f"风险等级: {result.risk_level} (得分: {result.total_risk_score})")


def cmd_query(args):
    """执行 RAG 查询"""
    engine = RAGEngine()
    if args.build:
        print("正在构建索引...")
        engine.build_index()

    if args.question:
        print(f"查询: {args.question}")
        result = engine.query(args.question)
        print(f"\n回答: {result.answer}")
        if result.sources:
            print("\n来源:")
            for s in result.sources:
                print(f" - {s.get('source', '未知')}")


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 系统命令行工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # process 子命令
    subparsers.add_parser("process", help="运行 OCR 和文档分类")

    # extract 子命令
    parser_extract = subparsers.add_parser("extract", help="提取风险实体")
    parser_extract.add_argument(
        "--input", default="docs/all_extracted.txt", help="输入文本文件路径"
    )
    parser_extract.add_argument(
        "--output", default="docs/entities_extracted.json", help="输出 JSON 文件路径"
    )

    # query 子命令
    parser_query = subparsers.add_parser("query", help="执行 RAG 查询")
    parser_query.add_argument("question", nargs="?", help="要查询的问题")
    parser_query.add_argument("--build", action="store_true", help="在查询前重新构建索引")

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
