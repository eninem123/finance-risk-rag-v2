"""
Finance-Risk-RAG 统一命令行入口
==============================
"""

import argparse
import logging
from pathlib import Path

from src.finance_risk_rag.config import get_config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.processor import DocumentProcessor
from src.finance_risk_rag.service import RiskAnalysisService
from src.finance_risk_rag.utils import setup_logger

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Finance-Risk-RAG: 银行级财务文本风控 AI 系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # Process 子命令
    process_parser = subparsers.add_parser("process", help="处理 PDF 文档 (OCR + 分类)")
    process_parser.add_argument("--dir", type=str, help="文档目录")
    process_parser.add_argument("--workers", type=int, default=4, help="并行工作进程数")

    # Extract 子命令
    extract_parser = subparsers.add_parser("extract", help="从文本中提取风险实体")
    extract_parser.add_argument(
        "--input", type=str, default="docs/all_extracted.txt", help="输入文本文件"
    )
    extract_parser.add_argument(
        "--output", type=str, default="docs/entities_extracted.json", help="输出 JSON 文件"
    )

    # Query 子命令
    query_parser = subparsers.add_parser("query", help="执行 RAG 风险问答")
    query_parser.add_argument("question", type=str, help="用户问题")
    query_parser.add_argument("--build", action="store_true", help="强制重建索引")
    query_parser.add_argument("--top-k", type=int, default=4, help="检索块数量")

    # Report 子命令
    report_parser = subparsers.add_parser("report", help="为 PDF 生成综合风险报告")
    report_parser.add_argument("--input", type=str, required=True, help="PDF 文件或目录路径")
    report_parser.add_argument("--output-dir", type=str, default="reports", help="报告保存目录")

    args = parser.parse_args()
    config = get_config()
    setup_logger("finance_risk_rag")

    if args.command == "process":
        processor = DocumentProcessor(config)
        docs_dir = Path(args.dir) if args.dir else config.docs_dir
        logger.info(f"正在处理目录: {docs_dir}")
        processor.process_directory(docs_dir, max_workers=args.workers)
        print(f"✅ 文档处理完成。结果已保存至 {docs_dir}")

    elif args.command == "extract":
        pipeline = EntityExtractionPipeline(config)
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ 错误: 输入文件 {args.input} 不存在。")
            return

        result = pipeline.process(input_path)
        from src.finance_risk_rag.utils import save_json_file

        save_json_file(result.to_dict(), Path(args.output))
        print(f"✅ 实体提取完成。风险等级: {result.risk_level}, 总分: {result.total_risk_score}")
        print(f"结果已保存至 {args.output}")

    elif args.command == "query":
        engine = RAGEngine(config)
        if args.build:
            print("正在构建索引...")
            engine.build_index()

        result = engine.query(args.question, top_k=args.top_k)
        print(f"\n🔍 问题: {args.question}")
        print(f"\n🤖 回答: {result.answer}")
        print("\n📚 来源:")
        for i, src in enumerate(result.sources):
            print(f"  [{i+1}] {src.get('source')} (块 {src.get('chunk_index')})")

    elif args.command == "report":
        service = RiskAnalysisService(config)
        input_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            if input_path.suffix.lower() != ".pdf":
                print("❌ 错误: 仅支持 PDF 文件。")
                return

            analysis_data = service.run_full_analysis(input_path)
            report_path = output_dir / f"Risk_Report_{input_path.stem}.md"
            service.generate_report(analysis_data, report_path)
            print(f"✅ 报告已生成: {report_path}")

        elif input_path.is_dir():
            print(f"正在批量分析目录: {input_path}")
            pdf_files = list(input_path.glob("*.pdf"))
            for pdf in pdf_files:
                try:
                    analysis_data = service.run_full_analysis(pdf)
                    report_path = output_dir / f"Risk_Report_{pdf.stem}.md"
                    service.generate_report(analysis_data, report_path)
                    print(f"  - [{pdf.name}] -> {report_path.name}")
                except Exception as e:
                    print(f"  - [{pdf.name}] ❌ 失败: {e}")
            print(f"✅ 批量处理完成。共处理 {len(pdf_files)} 个文件。")
        else:
            print(f"❌ 错误: 路径 {args.input} 无效。")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
