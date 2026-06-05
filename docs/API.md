# API Documentation

## `src.finance_risk_rag.engine`

### `RAGEngine`
Main class for RAG operations.
- `__init__(config: Optional[Config] = None)`: Initializes the engine with configuration, ChromaDB client, and embedding function (ONNX MiniLM).
- `build_index(docs_dir: Optional[Path] = None) -> Dict[str, int]`: Indexes text files into the vector database. Returns statistics about processed files and chunks.
- `query(question: str, top_k: int = 5) -> QueryResult`: Performs retrieval-augmented generation to answer questions. Includes source attribution.

## `src.finance_risk_rag.processor`

### `DocumentProcessor`
Handles document OCR and classification.
- `process_batch(docs_dir: Optional[Path] = None)`: Processes all PDFs in the given directory. Uses MD5 hashing and versioning to avoid redundant processing.
- `extract_text_from_pdf(pdf_path: Path) -> Tuple[str, int]`: Low-level PDF extraction with OCR fallback for image-heavy pages.
- `classify_document(text_sample: str) -> Dict[str, Any]`: Uses LLM to categorize documents into types like "Audit Report" or "Financial Statement".

## `src.finance_risk_rag.extractor`

### `EntityExtractionPipeline`
Coordinated pipeline for risk analysis.
- `process(text: str) -> ExtractionResult`: Combines rule-based and BERT-based extraction to identify risk entities and calculate an overall risk score.

### `RuleBasedExtractor`
- `extract(text: str) -> List[Entity]`: Uses regex and keyword matching based on rules in `knowledge_base/risk_entities.json`.

### `BERTExtractor`
- `extract(text: str) -> List[Entity]`: Skeleton for transformer-based Named Entity Recognition (NER).

## `src.finance_risk_rag.llm`

### `LLMClientWrapper`
Unified interface for LLM providers.
- `chat(messages: List[Dict[str, str]]) -> str`: Standard chat interface.
- `ask(query: str, context: str) -> str`: High-level interface for context-aware Q&A.

## `src.finance_risk_rag.models`
Centralized dataclasses for data integrity:
- `Entity`: Individual risk findings.
- `ExtractionResult`: Aggregated report for a document.
- `QueryResult`: RAG answer with sources.
- `ChunkConfig`: Text splitting parameters.
- `DocumentChunk`: Atomic unit for vector storage.
