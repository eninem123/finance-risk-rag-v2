"""
Finance-Risk-RAG 统一命令行入口
==============================
"""

import argparse
import sys
from pathlib import Path

# 动态添加 src 到 sys.path
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.engine import RAGEngine  # noqa: E402
from finance_risk_rag.extractor import EntityExtractionPipeline  # noqa: E402
from finance_risk_rag.processor import DocumentProcessor  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from finance_risk_rag.utils import setup_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG: 银行级财务文本风控 AI 系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # Process 子命令
    process_parser = subparsers.add_parser("process", help="处理 PDF 文档 (OCR + 分类)")
    process_parser.add_argument("--dir", type=str, help="文档目录")

    # Extract 子命令
    extract_parser = subparsers.add_parser("extract", help="提取风险实体")
    extract_parser.add_argument(
        "--input", type=str, default="docs/all_extracted.txt", help="输入文本文件"
    )
    extract_parser.add_argument(
        "--output", type=str, default="docs/entities_extracted.json", help="输出 JSON 文件"
    )

    # Query 子命令
    query_parser = subparsers.add_parser("query", help="执行 RAG 问答")
    query_parser.add_argument("question", type=str, help="用户问题")
    query_parser.add_argument("--build", action="store_true", help="先构建索引")

    # Report 子命令
    report_parser = subparsers.add_parser("report", help="生成全流程风险报告")
    report_parser.add_argument("--dir", type=str, help="文档目录")

    args = parser.parse_args()
    config = get_config()
    setup_logger("finance_risk_rag")

    if args.command == "process":
        processor = DocumentProcessor(config)
        docs_dir = Path(args.dir) if args.dir else config.docs_dir
        processor.process_directory(docs_dir)
        print("文档处理完成。")

    elif args.command == "extract":
        pipeline = EntityExtractionPipeline(config)
        result = pipeline.process(Path(args.input))
        # Save result
        from finance_risk_rag.utils import save_json_file

        save_json_file(result.to_dict(), Path(args.output))
        print(f"实体提取完成。风险等级: {result.risk_level}, 总分: {result.total_risk_score}")

    elif args.command == "query":
        engine = RAGEngine(config)
        if args.build:
            print("正在构建索引...")
            engine.build_index()

        result = engine.query(args.question)
        print(f"\n回答: {result.answer}")
        print(f"\n来源: {result.sources}")

    elif args.command == "report":
        service = RiskAnalysisService(config)
        docs_dir = Path(args.dir) if args.dir else config.docs_dir
        print(f"正在分析目录 {docs_dir} 下的文档...")
        reports = service.batch_analyze(docs_dir)
        print(f"全流程报告生成完成，共处理 {len(reports)} 个文档。")
        print(f"报告已保存至 {docs_dir / 'comprehensive_risk_report.json'}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
