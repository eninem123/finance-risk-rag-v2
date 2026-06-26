import sys
from pathlib import Path

import pandas as pd

# Add src to sys.path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st  # noqa: E402

from src.finance_risk_rag.config import get_config  # noqa: E402
from src.finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from src.finance_risk_rag.utils import load_json_file  # noqa: E402

st.set_page_config(
    page_title="Finance-Risk-RAG v2.3 Enterprise Dashboard",
    page_icon="🏦",
    layout="wide",
)

config = get_config()

# Initialize service
if "service" not in st.session_state:
    st.session_state.service = RiskAnalysisService(config)

service = st.session_state.service

st.sidebar.title("🏦 Finance-Risk-RAG v2.3")
st.sidebar.markdown("银行级财务文本风控系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "文档分析", "风险检索"],
)

if page == "数据总览":
    st.title("📊 风险监控概览")

    # Load classification data
    class_path = config.docs_dir / "classification.json"
    if class_path.exists():
        data = load_json_file(class_path)
        if data:
            df = pd.DataFrame.from_dict(data, orient="index")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("文档类型分布")
                type_counts = df["type"].value_counts()
                st.bar_chart(type_counts)

            with col2:
                st.subheader("风险分控明细")
                st.dataframe(df[["type", "confidence"]], use_container_width=True)
        else:
            st.info("暂无处理数据，请先通过 CLI 处理文档。")
    else:
        st.info("请先执行文档处理：`python main.py process`")

elif page == "文档分析":
    st.title("📑 文档风险深度分析")

    # List processed text files
    txt_files = list(config.docs_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

    if not txt_files:
        st.warning("未发现处理后的文档。请先在 CLI 运行 `process`。")
    else:
        selected_file = st.selectbox("选择待审阅文档", [f.name for f in txt_files])
        txt_path = config.docs_dir / selected_file

        if st.button("生成风控专家建议"):
            with st.spinner("风控 AI 正在深度扫描文档内容..."):
                result = service.pipeline.process(txt_path)

                # Executive Summary Section
                st.subheader("📋 风险执行摘要")
                summary = service._generate_executive_summary(
                    result.risk_level,
                    [e.to_dict() for e in result.entities]
                )
                st.info(summary)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("综合风险等级", result.risk_level)
                m2.metric("量化风险总分", result.total_risk_score)
                m3.metric("识别风险实体", len(result.entities))

                # Detailed Breakdown
                st.subheader("🚩 识别出的关键风险点")
                if result.entities:
                    entities_df = pd.DataFrame([e.to_dict() for e in result.entities])
                    # Reorder and rename columns for professional display
                    cols = ["type", "text", "risk_score", "confidence", "source"]
                    display_df = entities_df[cols].copy()
                    display_df.columns = ["类别", "实体文本", "权重", "置信度", "来源"]
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.success("文档内容合规，未发现显着财务风险实体。")

elif page == "风险检索":
    st.title("🔍 风险知识库检索 (RAG)")
    st.markdown("基于已索引文档的自然语言问答，用于调取特定风险条款或历史背景。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("例如：这笔贷款涉及哪些抵押资产？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("专家正在检索知识库..."):
                try:
                    res = service.query_risk(prompt)
                    response = res.answer
                    st.markdown(response)

                    if res.sources:
                        with st.expander("查看风控依据 (Source Context)"):
                            for src in res.sources:
                                st.write(f"- 来源文件: `{src.get('source', '未知')}` (块索引: {src.get('chunk_index')})")
                except Exception as e:
                    response = f"检索失败: {e}"
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
