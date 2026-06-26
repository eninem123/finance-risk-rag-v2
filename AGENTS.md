# Agent Guidelines for Finance-Risk-RAG

Welcome to the Finance-Risk-RAG project. As an AI assistant working on this codebase, you should adhere to the following principles and standards.

## 🏛 Architectural Principles

1.  **Service-Oriented Orchestration**: Always use `RiskAnalysisService` as the entry point for business logic. Avoid instantiating low-level components like `DocumentProcessor` or `BERTExtractor` directly in CLI or UI layers.
2.  **Type Safety with Pydantic**: All data structures must be defined as Pydantic models in `models.py`. All configurations must use `Config` in `config.py`.
3.  **Domain-Specific Exceptions**: Use the custom exceptions defined in `exceptions.py`. Avoid broad `try-except Exception` blocks unless logging and re-raising a domain exception.
4.  **OOP & DI**: Follow Object-Oriented Programming and Dependency Injection. Components should accept `config` and `llm_client` in their `__init__`.

## 💻 Coding Standards

-   **Line Length**: 100 characters max.
-   **Formatting**: Use Black and Isort.
-   **Type Hints**: Mandatory for all public methods and functions.
-   **Docstrings**: Follow Google-style docstrings.
-   **Logging**: Use the system-wide logger. Log significant events (IO, API calls, errors) at appropriate levels.

## 🧪 Testing Requirements

-   All new features must include unit tests in the `tests/` directory.
-   Run the full test suite using `pytest tests/` before proposing any PR.
-   Mock external services (LLM, OCR) in unit tests.

## 📝 Change Management

-   Update `CHANGELOG.md` (if exists) for significant changes.
-   Increment the version number in `README.md`, `main.py`, and `dashboard.py` when releasing new core features.
-   Maintain backward compatibility for the `RiskAnalysisService` public API.

## 🏦 Financial Context

Remember that this is a bank-grade system. Prioritize:
-   **Auditability**: Every step of the analysis should be traceable.
-   **Robustness**: Handle malformed PDFs and API timeouts gracefully.
-   **Professionalism**: UI and reports should use formal, professional financial terminology.
