import sys
from pathlib import Path

import pandas as pd

# Add src to sys.path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.finance_risk_rag.config import get_config  # noqa: E402
from src.finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from src.finance_risk_rag.utils import load_json_file  # noqa: E402

st.set_page_config(
    page_title="Finance-Risk-RAG v2.3 Dashboard",
    page_icon="🏦",
    layout="wide",
)

config = get_config()

# Initialize service
if "service" not in st.session_state:
    st.session_state.service = RiskAnalysisService(config)

service = st.session_state.service


@st.cache_data(show_spinner=False)
def get_cached_extraction(_service, txt_path):
    """缓存实体提取结果，避免重复运行 NLP 模型"""
    return _service.pipeline.process(txt_path)

st.sidebar.title("🏦 Finance-Risk-RAG v2.3")
st.sidebar.markdown("银行级多语言财务文本风控系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "文档分析", "风险检索", "风险报告"],
)

if page == "数据总览":
    st.title("📊 数据总览")

    # Load classification data
    class_path = config.docs_dir / "classification.json"
    if class_path.exists():
        data = load_json_file(class_path)
        if data:
            df = pd.DataFrame.from_dict(data, orient="index")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("文档类型分布")
                type_counts = df["type"].value_counts()
                st.bar_chart(type_counts)

            with col2:
                st.subheader("处理详情")
                st.dataframe(df[["type", "confidence"]], use_container_width=True)

            # Risk Matrix (V2.3 Feature)
            st.markdown("---")
            st.subheader("🎯 风险矩阵 (Impact vs. Confidence)")

            # Aggregate risk entities for visualization
            entities_data = []
            txt_files = list(config.docs_dir.glob("*.txt"))
            txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

            for txt_f in txt_files[:5]:  # Limit to top 5 files for dashboard performance
                res = get_cached_extraction(service, txt_f)
                for e in res.entities:
                    entities_data.append(
                        {
                            "text": e.text,
                            "type": e.type,
                            "risk_score": e.risk_score,
                            "confidence": e.confidence,
                            "impact_score": e.impact_score,
                        }
                    )

            if entities_data:
                edf = pd.DataFrame(entities_data)
                fig = px.scatter(
                    edf,
                    x="confidence",
                    y="impact_score",
                    size="risk_score",
                    color="type",
                    hover_name="text",
                    labels={"confidence": "置信度", "impact_score": "影响权重"},
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无处理数据。")
    else:
        st.info("请先执行文档处理。")

elif page == "文档分析":
    st.title("📑 文档分析")

    # List processed text files
    txt_files = list(config.docs_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

    if not txt_files:
        st.warning("未发现处理后的文档文本。")
    else:
        selected_file = st.selectbox("选择文档", [f.name for f in txt_files])
        txt_path = config.docs_dir / selected_file

        if st.button("开始提取风险实体"):
            with st.spinner("正在提取..."):
                result = service.pipeline.process(txt_path)

                st.subheader(f"风险等级: {result.risk_level}")
                st.metric("总风险评分", result.total_risk_score)

                if result.entities:
                    entities_df = pd.DataFrame([e.to_dict() for e in result.entities])
                    st.dataframe(entities_df, use_container_width=True)
                else:
                    st.success("未检测到显著风险实体。")

elif page == "风险检索":
    st.title("🔍 风险检索 (RAG)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("输入关于财务文档的风险问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    res = service.query_risk(prompt)
                    response = res.answer
                    st.markdown(response)

                    if res.sources:
                        with st.expander("查看来源"):
                            for src in res.sources:
                                st.write(f"- {src.get('source', '未知')}")
                except Exception as e:
                    response = f"发生错误: {e}"
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "风险报告":
    st.title("📄 风险报告生成器")

    pdf_files = list(config.docs_dir.glob("*.pdf"))
    if not pdf_files:
        st.info("请先在 docs 目录下放置 PDF 文件。")
    else:
        selected_pdf = st.selectbox("选择 PDF 进行深入分析", [f.name for f in pdf_files])
        pdf_path = config.docs_dir / selected_pdf

        if st.button("一键生成分析报告"):
            with st.spinner("深度分析中..."):
                analysis = service.analyze_document(pdf_path)
                report_md = service.generate_report(analysis)

                st.success("分析完成！")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("报告预览")
                    st.markdown(report_md)

                with col2:
                    st.subheader("操作")
                    st.download_button(
                        label="下载 Markdown 报告",
                        data=report_md,
                        file_name=f"{pdf_path.stem}_risk_report.md",
                        mime="text/markdown",
                    )

                    st.metric("风险等级", analysis["risk_analysis"]["risk_level"])
                    st.metric("量化总分", analysis["risk_analysis"]["total_risk_score"])
