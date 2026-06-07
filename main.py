import argparse
import logging
from pathlib import Path

from finance_risk_rag.engine import RAGEngine
from finance_risk_rag.extractor import EntityExtractionPipeline
from finance_risk_rag.processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")


def cmd_process(args):
    processor = DocumentProcessor()
    stats = processor.batch_process()
    print(f"处理完成: {stats}")


def cmd_extract(args):
    pipeline = EntityExtractionPipeline()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件 {input_path} 不存在")
        return

    result = pipeline.process(input_path)
    output_path = Path(args.output)
    pipeline.save_result(result, output_path)

    print("实体提取完成!")
    print(f"  实体数: {len(result.entities)}")
    print(f"  总风险评分: {result.total_risk_score} ({result.risk_level})")
    print(f"  结果已保存至: {output_path}")


def cmd_query(args):
    engine = RAGEngine()
    if args.build:
        print("正在构建/更新索引...")
        stats = engine.build_index()
        print(f"索引构建完成: {stats}")

    if args.question:
        print(f"查询中: {args.question}")
        result = engine.query(args.question, top_k=args.top_k)
        print("\n" + "=" * 50)
        print(f"回答: {result.answer}")
        print("=" * 50)
        if result.sources:
            print("\n来源:")
            for src in result.sources:
                print(f"  - {src.get('source')} (分块 {src.get('chunk_index')})")


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG: 银行级多语言财务文本风控AI系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # Process subcommand
    subparsers.add_parser("process", help="PDF文本提取与分类")

    # Extract subcommand
    parser_extract = subparsers.add_parser("extract", help="风险实体提取")
    parser_extract.add_argument(
        "--input", type=str, default="docs/all_extracted.txt", help="输入文本路径"
    )
    parser_extract.add_argument(
        "--output", type=str, default="docs/entities_extracted.json", help="输出JSON路径"
    )

    # Query subcommand
    parser_query = subparsers.add_parser("query", help="RAG问答查询")
    parser_query.add_argument("question", nargs="?", type=str, help="用户问题")
    parser_query.add_argument("--build", action="store_true", help="构建向量数据库索引")
    parser_query.add_argument("--top-k", type=int, default=4, help="检索相关文档块数量")

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
