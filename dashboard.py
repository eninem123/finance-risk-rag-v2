import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.finance_risk_rag.config import get_config
from src.finance_risk_rag.service import RiskAnalysisService

st.set_page_config(page_title="Finance-Risk-RAG Dashboard", layout="wide")

st.title("🏦 Finance-Risk-RAG: 智能财务风控看板")

config = get_config()
service = RiskAnalysisService(config)

with st.sidebar:
    st.header("⚙️ 系统配置")
    docs_dir = st.text_input("文档目录", value=str(config.docs_dir))
    if st.button("🚀 开始全量分析"):
        with st.spinner("正在分析中，请稍候..."):
            service.analyze_directory(Path(docs_dir))
            st.success("分析完成！")

st.header("📊 风险概览")

report_path = Path(docs_dir) / "risk_report.json"
if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    summary = report_data["summary"]

    col1, col2, col3 = st.columns(3)
    col1.metric("风险等级", summary["risk_level"])
    col2.metric("总风险分数", summary["total_risk_score"])
    col3.metric("实体总数", summary["total_entities"])

    st.subheader("🔍 风险实体列表")
    df = pd.DataFrame(summary["entities"])
    if not df.empty:
        st.dataframe(
            df[["type", "text", "risk_score", "confidence", "source"]],
            use_container_width=True,
        )

    st.subheader("💡 风险问答 (RAG)")
    question = st.text_input(
        "针对本项目提出你的疑问：", placeholder="例如：这笔贷款的主要风险点在哪里？"
    )
    if question:
        with st.spinner("思考中..."):
            res = service.engine.query(question)
            st.write("**回答:**")
            st.write(res.answer)
            with st.expander("查看参考来源"):
                st.write(res.sources)
else:
    st.info("请先点击侧边栏的按钮开始分析，或确保指定目录下存在 `risk_report.json`。")

st.markdown("---")
st.caption("Powered by Finance-Risk-RAG v2.1 | 银行级多语言财务文本风控 AI 系统")
