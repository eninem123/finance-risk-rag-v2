# Development Guide

## Environment Setup
1. **Python Version**: 3.9+ is required. 3.12 is recommended for performance.
2. **Dependencies**: `pip install -r requirements.txt`.
3. **External Tools**:
   - **Tesseract OCR**: Required for processing image-based PDFs. Set the `TESSERACT_CMD` environment variable to the executable path.
   - **ChromaDB**: Used as the vector database, initialized automatically in `rag_db/`.

## Project Architecture
The project follows a modular, Object-Oriented design:
- `main.py`: Entry point for all CLI operations.
- `src/finance_risk_rag/`: Core package containing domain logic.
- `models.py`: Centralized source of truth for data structures.
- `config.py`: Centralized configuration with environment variable support.
- `processor.py`: PDF parsing, OCR optimization, and document classification.
- `extractor.py`: Hybrid entity extraction (Rules + BERT).
- `engine.py`: Vector search and RAG orchestration.

## Coding Standards
- **Formating**: All code should be formatted with `Black` (line length 100).
- **Import Sorting**: Use `isort` with the `black` profile.
- **Linting**: `flake8` is used for static analysis.
- **Type Hints**: Type hinting is mandatory for all new functions and classes.

## Testing
Comprehensive unit tests are located in the `tests/` directory.
Run tests using:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m pytest tests/
```

## How to Extend
- **New Risk Types**: Add keywords and scores to `knowledge_base/risk_entities.json`.
- **New LLM Providers**: Update `LLMClientWrapper` in `llm.py` to support additional OpenAI-compatible APIs.
- **OCR Tweaks**: Modify `optimize_image` in `processor.py` to improve recognition for specific document types.
