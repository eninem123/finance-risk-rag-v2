import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 确保可以导入 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from finance_risk_rag.engine import RAGEngine  # noqa: E402

st.set_page_config(page_title="Finance-Risk-RAG Dashboard", layout="wide")

st.title("🏦 Finance-Risk-RAG 财务风控智能看板")
st.markdown("---")

config = get_config()
service = RiskAnalysisService(config)

# Sidebar
st.sidebar.header("⚙️ 系统配置")
st.sidebar.info(f"LLM Provider: {config.llm_provider}\n\nModel: {config.llm_model_name}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["📊 文档分析", "🔍 实体可视化", "🧠 RAG 问答"])

with tab1:
    st.header("📂 文档风控分析")
    uploaded_file = st.file_uploader("上传 PDF 财务文档", type="pdf")

    if uploaded_file:
        temp_path = Path("docs") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("开始全面分析"):
            with st.spinner("正在执行 OCR、分类与实体提取..."):
                report = service.analyze_document(temp_path)
                st.success("分析完成！")

                col1, col2, col3 = st.columns(3)
                col1.metric("文档类型", report["classification"]["type"])
                col2.metric("风险评分", report["risk_assessment"]["score"])
                col3.metric("风险等级", report["risk_assessment"]["level"])

                st.subheader("📝 智能摘要")
                st.write(report["ai_analysis"]["summary"])

                st.subheader("🚩 风险实体详情")
                entities = report["risk_assessment"]["entities"]
                if entities:
                    df = pd.DataFrame(entities)
                    st.dataframe(df[["type", "text", "risk_score", "source", "context"]])
                else:
                    st.info("未发现明显风险实体。")

with tab2:
    st.header("🔍 风险实体全局视图")
    class_file = Path("docs/classification.json")
    if class_file.exists():
        with open(class_file, "r", encoding="utf-8") as f:
            classes = json.load(f)
        st.subheader("文档分类分布")
        df_class = pd.DataFrame([{"Document": k, **v} for k, v in classes.items()])
        st.dataframe(df_class)
    else:
        st.info("请先通过 CLI 或文档分析功能处理文档。")

with tab3:
    st.header("🧠 知识库问答 (RAG)")
    engine = RAGEngine(config)

    question = st.text_input(
        "针对已处理的文档提出问题：", placeholder="例如：该公司的流动比率是否存在异常？"
    )

    if question:
        with st.spinner("检索并生成回答中..."):
            result = engine.query(question)
            st.markdown(f"### 🤖 AI 回答\n{result.answer}")

            with st.expander("查看来源上下文"):
                for i, src in enumerate(result.sources):
                    st.markdown(f"**来源 {i+1}:** {src.get('source', 'Unknown')}")
                    st.text(src.get("content", "")[:500] + "...")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Finance-Risk-RAG v2.2")
