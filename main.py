"""
Finance-Risk-RAG 主入口
========================
"""

import argparse
import sys
from pathlib import Path

# Add src to path if needed, though usually handled by environment
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag import (
    DocumentProcessor,
    ExtractionPipeline,
    RAGEngine,
    get_config,
)


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG Optimized Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process documents
    subparsers.add_parser("process", help="Extract text from PDFs")

    # Extract entities
    ext_parser = subparsers.add_parser("extract", help="Extract risk entities from text")
    ext_parser.add_argument(
        "--input", type=str, default="docs/all_extracted.txt", help="Input file"
    )

    # RAG Query
    rag_parser = subparsers.add_parser("query", help="Query the RAG system")
    rag_parser.add_argument("question", type=str, help="The question to ask")
    rag_parser.add_argument("--build", action="store_true", help="Rebuild index before query")

    args = parser.parse_args()
    config = get_config()

    if args.command == "process":
        print("Starting document processing...")
        processor = DocumentProcessor(config)
        stats = processor.batch_process()
        print(f"Processing complete: {stats}")

    elif args.command == "extract":
        print(f"Extracting entities from {args.input}...")
        pipeline = ExtractionPipeline()
        result = pipeline.run(Path(args.input))
        print(f"Extraction result: {result}")

    elif args.command == "query":
        engine = RAGEngine()
        if args.build:
            print("Building index...")
            engine.build_index()

        print(f"Querying: {args.question}")
        result = engine.query(args.question)
        print(f"\nAnswer: {result.answer}")
        print(f"\nSources: {result.sources}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
