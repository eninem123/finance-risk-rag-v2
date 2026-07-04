"""
Finance-Risk-RAG 统一命令行入口
==============================
"""

import argparse
import sys
from pathlib import Path

# Standardize sys.path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.finance_risk_rag.config import get_config  # noqa: E402
from src.finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from src.finance_risk_rag.utils import save_json_file, setup_logger  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Finance-Risk-RAG: 银行级财务文本风控 AI 系统 (v2.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    process_parser = subparsers.add_parser("process", help="全流程处理 PDF (OCR + 提取 + 索引)")
    process_parser.add_argument("--input", type=str, help="文档或目录路径")

    extract_parser = subparsers.add_parser("extract", help="从文本提取风险实体")
    extract_parser.add_argument(
        "--input", type=str, default="docs/all_extracted.txt", help="输入文本文件"
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        default="docs/entities_extracted.json",
        help="输出 JSON 文件",
    )

    query_parser = subparsers.add_parser("query", help="执行 RAG 风险问答")
    query_parser.add_argument("question", type=str, help="用户问题")
    query_parser.add_argument("--build", action="store_true", help="强制重建索引")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索块数量")

    report_parser = subparsers.add_parser("report", help="为 PDF 生成综合风险报告")
    report_parser.add_argument("--input", type=str, required=True, help="PDF 文件或目录路径")
    report_parser.add_argument("--output-dir", type=str, default="reports", help="报告保存目录")

    subparsers.add_parser("dashboard", help="启动 Streamlit 可视化面板")

    args = parser.parse_args()
    config = get_config()
    setup_logger("finance_risk_rag")

    service = RiskAnalysisService(config)

    if args.command == "process":
        input_path = Path(args.input) if args.input else config.docs_dir
        if not input_path.exists():
            print(f"错误: 路径不存在 - {input_path}")
            return

        print(f"正在分析: {input_path}")
        results = service.run_full_analysis(input_path)
        count = len(results) if isinstance(results, dict) else 0
        print(f"分析完成。处理了 {count} 个文件。")

    elif args.command == "extract":
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 文件不存在 - {input_path}")
            return

        result = service.pipeline.process(input_path)
        save_json_file(result.to_dict(), Path(args.output))
        print(f"✅ 实体提取完成。风险等级: {result.risk_level}, 总分: {result.total_risk_score}")
        print(f"结果已保存至 {args.output}")

    elif args.command == "query":
        if args.build:
            print("正在构建索引...")
            service.engine.build_index()

        result = service.query_risk(args.question)
        print(f"\n回答: {result.answer}")
        print(f"\n来源: {result.sources}")

    elif args.command == "report":
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        if not input_path.exists():
            print(f"错误: 路径不存在 - {input_path}")
            return

        if input_path.is_file():
            analysis = service.analyze_document(input_path)
            report_path = output_dir / f"{input_path.stem}_report.md"
            service.generate_report(analysis, report_path)
            print(f"✅ 报告已生成: {report_path}")
        else:
            batch_results = service.process_batch(input_path)
            for analysis in batch_results:
                name = analysis["document_info"]["name"]
                report_path = output_dir / f"{Path(name).stem}_report.md"
                service.generate_report(analysis, report_path)
            print(f"✅ 批量报告已生成 {len(batch_results)} 份，目录: {output_dir}")

    elif args.command == "dashboard":
        import os

        os.system("streamlit run dashboard.py")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
