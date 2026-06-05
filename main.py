"""
Finance-Risk-RAG Unified CLI.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.finance_risk_rag.config import get_config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractionPipeline
from src.finance_risk_rag.processor import DocumentProcessor
from src.finance_risk_rag.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG v2.0")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="OCR and classify documents")
    process_parser.add_argument("--docs-dir", type=str, help="Directory containing PDF files")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract risk entities")
    extract_parser.add_argument("--input", type=str, help="Input text file path")
    extract_parser.add_argument("--output", type=str, help="Output JSON file path")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.add_argument(
        "--build", action="store_true", help="Build/update index before querying"
    )
    query_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )

    args = parser.parse_args()

    # Configure logging
    config = get_config()
    setup_logger("finance_risk_rag", log_file=config.log_dir / "app.log")
    logger = logging.getLogger("finance_risk_rag")

    if args.command == "process":
        processor = DocumentProcessor()
        docs_dir = Path(args.docs_dir) if args.docs_dir else None
        processor.process_batch(docs_dir)
        logger.info("Processing complete.")

    elif args.command == "extract":
        pipeline = EntityExtractionPipeline()
        input_path = Path(args.input) if args.input else config.docs_dir / "all_extracted.txt"
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            sys.exit(1)

        text = input_path.read_text(encoding="utf-8")
        result = pipeline.process(text)

        output_path = (
            Path(args.output) if args.output else config.docs_dir / "entities_extracted.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"Extraction complete. Risk level: {result.risk_level}")
        print(f"Results saved to: {output_path}")

    elif args.command == "query":
        engine = RAGEngine()
        if args.build:
            print("Building index...")
            stats = engine.build_index()
            print(f"Index built: {stats}")

        result = engine.query(args.question, top_k=args.top_k)
        print(f"\nQuestion: {args.question}")
        print(f"Answer: {result.answer}")
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.get('source')} (Index: {source.get('index')})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
