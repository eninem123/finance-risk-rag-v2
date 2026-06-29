.PHONY: install test lint dashboard clean help

PYTHON = python3
PIP = pip3

help:
	@echo "Finance-Risk-RAG 开发者工具"
	@echo "用法:"
	@echo "  make install         安装依赖"
	@echo "  make test            运行单元测试"
	@echo "  make lint            代码格式检查 (flake8, black)"
	@echo "  make format          代码自动格式化 (isort, black)"
	@echo "  make dashboard       启动 Streamlit 可视化面板"
	@echo "  make clean           清理缓存文件"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install pydantic pydantic-settings httpx openai plotly black isort flake8

test:
	export PYTHONPATH=$$PYTHONPATH:$$(pwd)/src && $(PYTHON) -m pytest tests/

lint:
	flake8 . --count --max-line-length=100 --statistics
	black --check --line-length=100 .

format:
	isort .
	black --line-length=100 .

dashboard:
	streamlit run dashboard.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf cache/*
	rm -rf rag_db/*
