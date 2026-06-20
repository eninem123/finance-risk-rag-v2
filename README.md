# Finance-Risk-RAG v2.2

<div align="center">

**Enterprise-Grade Multi-language Financial Risk Control AI System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.2-blue?style=flat-square)]()

Advanced OCR · Hybrid Entity Extraction · RAG-based Risk Q&A · Streamlit Analytics

</div>

---

## 🏛 System Architecture

Finance-Risk-RAG is a professional AI framework designed for financial institutions to automate the risk assessment of complex documents. It integrates multiple cutting-edge technologies into a unified pipeline:

1.  **Intelligent Document Processor**: Utilizing Tesseract OCR and `pdfplumber` to extract text from both digital and scanned PDFs, with advanced image preprocessing for high-accuracy financial table recognition.
2.  **Hybrid Extraction Engine**: Combines high-precision Rule-Based extractors (for known financial risk keywords) with BERT-based deep learning models (for context-aware entity discovery).
3.  **Enterprise Service Layer**: Orchestrates the entire workflow, providing a robust API for document classification, entity merging, and automated report generation.
4.  **RAG Analysis Engine**: Powered by ChromaDB and LLMs (OpenAI/Moonshot), enabling conversational risk analysis over indexed document repositories.

---

## 🚀 Key Capabilities

-   **High-Fidelity Extraction**: Precise character-offset tracking (`start_char`/`end_char`) for all identified risks.
-   **Intelligent Classification**: Automatic document categorization (e.g., Audit Reports, Industry Analysis) using LLM-driven zero-shot classification.
-   **Automated Risk Reporting**: Generates comprehensive Markdown and JSON reports with quantitative risk scores and prioritized action items.
-   **Parallel Batch Processing**: Optimized for enterprise workloads using `ProcessPoolExecutor` for high-throughput document processing.
-   **Interactive Dashboard**: A professional Streamlit interface for visualization, deep-dive analysis, and RAG-based search.

---

## 🛠 Quick Start

### Installation

```bash
git clone https://github.com/eninem123/finance-risk-rag-v2.git
cd finance-risk-rag-v2
pip install -r requirements.txt
```

### Usage

```bash
# 1. Full Pipeline Processing (OCR -> Extraction -> Indexing)
python main.py process --input ./docs/

# 2. Risk Q&A (RAG)
python main.py query "What are the major risk factors identified in the latest audit?"

# 3. Comprehensive Risk Report Generation
python main.py report --input sample_audit.pdf --output-dir reports/

# 4. Launch Interactive Analytics Dashboard
python main.py dashboard
```

---

## 📂 Project Structure

```text
finance-risk-rag-v2/
├── src/finance_risk_rag/
│   ├── service.py          # Enterprise Orchestration Layer
│   ├── extractor.py        # Hybrid Extraction Engine (Rule + BERT)
│   ├── processor.py        # Intelligent OCR & Document Processing
│   ├── engine.py           # RAG Retrieval & Knowledge Engine
│   ├── llm.py              # LLM Client Wrapper with Retry Logic
│   └── models.py           # Unified Data Models (Entity, ExtractionResult)
├── dashboard.py            # Streamlit Interactive Dashboard
├── main.py                 # Unified Command-Line Interface (CLI)
├── tests/                  # Professional Test Suite
└── knowledge_base/         # Financial Dictionaries & Rule Sets
```

---

## 🛡 Security & Compliance

The system is built with financial compliance in mind:
-   **Auditable Logs**: Every extraction and LLM call is logged for transparency.
-   **Local Processing**: Supports local BERT models and local vector stores to ensure data privacy.
-   **Customizable Rules**: Easily update the risk dictionary in `knowledge_base/` to match local regulatory requirements.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
