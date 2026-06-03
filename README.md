# Finance-Risk-RAG v2.1

<div align="center">

**Professional Multi-language Financial Risk AI Control System**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [CLI Usage](#-cli-usage) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [CLI Usage](#-cli-usage)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [License](#-license)

---

## Introduction

Finance-Risk-RAG is a professional-grade intelligent risk control system designed for bank-level financial document analysis. It integrates OCR, document classification, risk entity extraction, and Retrieval-Augmented Generation (RAG) to automate financial risk assessment and early warning.

### Key Use Cases

| Scenario | Efficiency Gain | Turnaround Time |
|----------|-----------------|-----------------|
| Pre-loan Review | 70% | 24 Hours → 10 Mins |
| Post-loan Monitoring | 85% | 3 Days → 30 Mins |
| Automated Warning | 92% | Manual → Real-time |

---

## ✨ Features

- 🏗️ **Modular Architecture**: Clean separation of concerns with `src/finance_risk_rag/`.
- 🔍 **Hybrid Extraction**: Combines rule-based keyword matching with AI-driven extraction.
- 🧠 **RAG-powered Q&A**: Domain-specific insights using ChromaDB and ONNX-optimized embeddings.
- 📄 **Advanced OCR**: High-fidelity text extraction from complex PDFs with image enhancement.
- 🛠️ **Professional CLI**: Unified command-line interface for all system operations.
- 🧪 **Comprehensive Testing**: Full test suite for core logic ensuring reliability.

---

## Architecture

The system is organized into a modular package for scalability and maintainability:

```
finance-risk-rag/
├── main.py                # Unified CLI Entry Point
├── src/
│   └── finance_risk_rag/  # Core Package
│       ├── config.py      # Configuration Management
│       ├── engine.py      # RAG & LLM Engine
│       ├── extractor.py   # Risk Entity Extraction
│       ├── processor.py   # Document & OCR Processing
│       └── utils.py       # Shared Utilities
├── tests/                 # Unit Test Suite
├── docs/                  # Document Storage
├── knowledge_base/        # Risk Rules & Dictionaries
└── rag_db/                # Vector Database
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 - 3.12
- Tesseract OCR (for scanned PDFs)
- OpenAI-compatible API Key (Moonshot AI or OpenAI)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/finance-risk-rag.git
cd finance-risk-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

---

## 💻 CLI Usage

The system provides a unified CLI via `main.py`.

### 1. Process Documents
Extract text and classify new PDF files in the `docs/` directory.
```bash
python main.py process
```

### 2. Extract Risk Entities
Perform detailed risk analysis on a specific text file.
```bash
python main.py extract --file docs/sample_report.txt
```

### 3. Query the System
Ask complex risk-related questions using the RAG engine.
```bash
python main.py query "What are the primary liquidity risks mentioned in the latest audit?" --index
```

---

## ⚙️ Configuration

System configuration is managed in `src/finance_risk_rag/config.py`. You can override any setting using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for LLM | (None) |
| `LLM_MODEL_NAME` | Model to use | `moonshot-v1-8k` |
| `CHUNK_SIZE` | RAG chunk size | `800` |
| `OCR_DPI` | OCR resolution | `600` |

---

## 🧪 Testing

We use `pytest` for unit testing. To run the full test suite:

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
