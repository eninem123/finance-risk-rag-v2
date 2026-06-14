"""
Finance-Risk-RAG 统一命令行入口
==============================
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径以方便导入
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
    report_parser = subparsers.add_parser("report", help="生成全面风险报告")
    report_parser.add_argument("--input", type=str, help="PDF 文件或目录")

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
        from src.finance_risk_rag.utils import save_json_file

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
        input_path = Path(args.input) if args.input else config.docs_dir

        if input_path.is_file():
            print(f"正在分析文档: {input_path.name}...")
            report = service.analyze_document(input_path)
            # 格式化输出简要报告
            print(f"\n--- 风险分析报告: {report['document']['name']} ---")
            print(
                f"文档类型: {report['document']['type']} (置信度: {report['document']['confidence']:.2f})"
            )
            print(f"风险等级: {report['risk_analysis']['risk_level']}")
            print(f"风险得分: {report['risk_analysis']['total_risk_score']}")
            print(f"发现实体数: {report['risk_analysis']['total_entities']}")
            print(f"摘要: {report['summary']}")
        else:
            print(f"正在批量处理目录: {input_path}...")
            reports = service.generate_batch_report(input_path)
            print(
                f"处理完成。共生成 {len(reports)} 份分析报告。结果已保存至 {input_path / 'risk_report_batch.json'}"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
