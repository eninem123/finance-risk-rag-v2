# Finance-Risk-RAG v2.3 Makefile
# ==========================================

.PHONY: help install test lint format dashboard

help:
	@echo "Finance-Risk-RAG 自动化管理脚本"
	@echo "--------------------------------"
	@echo "make install   - 安装核心依赖"
	@echo "make test      - 运行单元测试"
	@echo "make lint      - 执行 Flake8 代码检查"
	@echo "make format    - 使用 Black 和 Isort 格式化代码"
	@echo "make dashboard - 启动 Streamlit 面板"

install:
	pip install -r requirements.txt

test:
	export PYTHONPATH=$${PYTHONPATH}:$(shell pwd)/src && python3 -m pytest tests/

lint:
	flake8 . --count --max-line-length=100 --statistics

format:
	isort .
	black --line-length=100 .

dashboard:
	streamlit run dashboard.py
