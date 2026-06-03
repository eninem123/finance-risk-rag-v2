"""
Finance-Risk-RAG Unified CLI
============================

Primary entry point for the Finance-Risk-RAG system.
"""

import argparse
from pathlib import Path

from src.finance_risk_rag.config import get_config
from src.finance_risk_rag.engine import RAGEngine
from src.finance_risk_rag.extractor import EntityExtractor
from src.finance_risk_rag.processor import DocumentProcessor
from src.finance_risk_rag.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG: Professional Financial Risk AI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Process Command
    process_parser = subparsers.add_parser(
        "process", help="Process PDF documents (OCR + Classification)"
    )
    process_parser.add_argument("--dir", type=str, help="Directory containing PDFs")

    # Extract Command
    extract_parser = subparsers.add_parser("extract", help="Extract risk entities from text files")
    extract_parser.add_argument(
        "--file", type=str, required=True, help="Text file to extract entities from"
    )
    extract_parser.add_argument("--output", type=str, help="Path to save extraction results (JSON)")

    # Query Command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.add_argument(
        "--index", action="store_true", help="Index documents before querying"
    )

    args = parser.parse_args()
    config = get_config()
    setup_logger("CLI")

    if args.command == "process":
        processor = DocumentProcessor(config)
        docs_dir = Path(args.dir) if args.dir else None
        results = processor.process_batch(docs_dir)
        print(f"Processed {len(results)} new documents.")

    elif args.command == "extract":
        extractor = EntityExtractor(config)
        text_path = Path(args.file)
        if not text_path.exists():
            print(f"Error: File {args.file} not found.")
            return

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        entities = extractor.extract(text)
        summary = extractor.summarize_risk(entities)

        if args.output:
            extractor.save_results(summary, Path(args.output))
        else:
            default_output = text_path.with_name("entities_extracted.json")
            extractor.save_results(summary, default_output)

        print(f"Found {summary['entity_count']} risk entities.")
        print(f"Total Risk Score: {summary['total_score']}")

    elif args.command == "query":
        engine = RAGEngine(config)
        if args.index:
            print("Indexing documents from docs directory...")
            engine.build_index()

        result = engine.query(args.question)
        print("\n--- AI Answer ---")
        print(result.answer)
        print("\n--- Sources ---")
        for src in result.sources:
            print(f"- {src.get('source')} (Chunk {src.get('chunk_index')})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
