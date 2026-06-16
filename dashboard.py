import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from finance_risk_rag.utils import load_json_file  # noqa: E402

st.set_page_config(page_title="Finance-Risk-RAG Dashboard", layout="wide")

st.title("🏦 Finance-Risk-RAG v2.2")
st.subheader("银行级多语言财务文本风控 AI 系统")

config = get_config()
service = RiskAnalysisService(config)

tabs = st.tabs(["📊 概览", "📑 文档分析", "🔍 智能问答", "⚙️ 配置"])

with tabs[0]:
    st.header("系统概览")
    col1, col2, col3 = st.columns(3)

    # 加载分类结果
    class_res = load_json_file(config.docs_dir / "classification.json")
    col1.metric("已处理文档", len(class_res))

    # 模拟一些统计数据
    col2.metric("识别风险实体", "124")
    col3.metric("平均风险分", "42.5")

    if class_res:
        df = pd.DataFrame(
            [
                {"文件名": k, "类型": v["type"], "置信度": v["confidence"]}
                for k, v in class_res.items()
            ]
        )
        st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.header("文档风控分析")
    pdf_files = list(config.docs_dir.glob("*.pdf"))
    selected_pdf = st.selectbox("选择文档进行分析", [f.name for f in pdf_files])

    if st.button("开始分析") and selected_pdf:
        with st.spinner("正在执行全流程分析..."):
            pdf_path = config.docs_dir / selected_pdf
            report = service.analyze_document(pdf_path)

            st.success("分析完成！")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.info(f"文档类型: {report['classification']['type']}")
                st.warning(f"风险等级: {report['risk_analysis']['risk_level']}")
                st.metric("总风险分", report["risk_analysis"]["total_risk_score"])

            with c2:
                st.write("**AI 总结:**")
                st.write(report["ai_summary"])

            st.write("**识别到的实体:**")
            entities = report["risk_analysis"]["entities"]
            if entities:
                st.table(pd.DataFrame(entities)[["type", "text", "risk_score", "source"]])

with tabs[2]:
    st.header("RAG 智能风险问答")
    question = st.text_input("请输入您关于财务风险的问题:")
    if st.button("咨询") and question:
        with st.spinner("检索知识库中..."):
            res = service.engine.query(question)
            st.write("**回答:**")
            st.write(res.answer)
            with st.expander("查看来源"):
                st.write(res.sources)

with tabs[3]:
    st.header("系统配置")
    st.json(
        {
            "LLM Provider": config.llm_provider,
            "Model": config.llm_model_name,
            "Chunk Size": config.chunk_size,
            "OCR DPI": config.ocr_dpi,
            "Risk Thresholds": {
                "Low": config.risk_level_low,
                "Medium": config.risk_level_medium,
                "High": config.risk_level_high,
            },
        }
    )
