"""
Finance-Risk-RAG 统一命令行入口
==============================
"""

import argparse
import sys
from pathlib import Path

# 确保 src 目录在路径中
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from finance_risk_rag.utils import setup_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Finance-Risk-RAG v2.2: 银行级财务文本风控 AI 系统"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # Process 子命令 (旧的，保留兼容性)
    process_parser = subparsers.add_parser("process", help="处理 PDF 文档 (OCR + 分类)")
    process_parser.add_argument("--dir", type=str, help="文档目录")

    # Report 子命令 (新的，推荐)
    report_parser = subparsers.add_parser("report", help="执行全流程分析并生成风险报告")
    report_parser.add_argument("--input", type=str, help="单个 PDF 路径或目录")

    # Query 子命令
    query_parser = subparsers.add_parser("query", help="执行 RAG 问答")
    query_parser.add_argument("question", type=str, help="用户问题")
    query_parser.add_argument("--build", action="store_true", help="先构建索引")

    args = parser.parse_args()
    config = get_config()
    setup_logger("finance_risk_rag")

    service = RiskAnalysisService(config)

    if args.command == "process":
        docs_dir = Path(args.dir) if args.dir else config.docs_dir
        service.processor.process_directory(docs_dir)
        print("文档处理完成。")

    elif args.command == "report":
        input_path = Path(args.input) if args.input else config.docs_dir
        if input_path.is_file():
            report = service.analyze_document(input_path)
            print(f"分析完成: {report['summary']}")
        else:
            reports = service.run_batch_analysis(input_path)
            print(f"批量分析完成，生成 {len(reports)} 份报告。")

    elif args.command == "query":
        if args.build:
            print("正在构建索引...")
            service.engine.build_index()

        result = service.query_risk(args.question)
        print(f"\n回答: {result['answer']}")
        print(f"\n来源: {result['sources']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
