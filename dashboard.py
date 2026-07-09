import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to sys.path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

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

st.sidebar.title("🏦 Finance-Risk-RAG v2.3")
st.sidebar.markdown("银行级多语言财务文本风控系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "文档分析", "风险矩阵", "风险检索"],
)

@st.cache_data
def get_cached_extraction(file_path):
    return service.pipeline.process(file_path)

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
                result = get_cached_extraction(txt_path)

                st.subheader(f"风险等级: {result.risk_level}")
                col1, col2 = st.columns(2)
                col1.metric("总风险评分", result.total_risk_score)
                col2.metric("识别实体数", len(result.entities))

                if result.entities:
                    entities_df = pd.DataFrame([e.to_dict() for e in result.entities])

                    st.subheader("详细风险清单")
                    st.dataframe(entities_df, use_container_width=True)

                    # 影响分分布图
                    st.subheader("风险影响分分布")
                    fig = px.bar(entities_df, x="text", y="impact_score", color="type",
                                 title="实体影响分统计", labels={"impact_score": "影响分数", "text": "实体"})
                    st.plotly_chart(fig, use_container_width=True)

                    # 报告预览与下载
                    st.divider()
                    st.subheader("📑 风险报告预览")

                    # 构造模拟的 analysis_data 供报告生成
                    analysis_data = {
                        "document_info": {
                            "name": selected_file.replace(".txt", ".pdf"),
                            "analyzed_at": result.extraction_time,
                        },
                        "classification": {"type": "未知", "confidence": 0.0, "reason": "Dashboard analysis"},
                        "risk_analysis": result.to_dict()
                    }

                    report_md = service.generate_report(analysis_data)
                    st.markdown(report_md)

                    st.download_button(
                        label="下载 Markdown 报告",
                        data=report_md,
                        file_name=f"Risk_Report_{selected_file.replace('.txt', '')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.success("未检测到显著风险实体。")

elif page == "风险矩阵":
    st.title("⚖️ 风险分析矩阵")

    txt_files = list(config.docs_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

    all_entities = []
    for txt in txt_files:
        res = get_cached_extraction(txt)
        for e in res.entities:
            d = e.to_dict()
            d["source_doc"] = txt.name
            all_entities.append(d)

    if all_entities:
        df = pd.DataFrame(all_entities)

        fig = px.scatter(df, x="confidence", y="impact_score", size="risk_score", color="type",
                         hover_name="text", facet_col="source_doc",
                         title="风险矩阵：置信度 vs 影响分",
                         labels={"confidence": "提取置信度", "impact_score": "风险影响分"})

        # 添加象限参考线
        fig.add_hline(y=df["impact_score"].mean(), line_dash="dash", annotation_text="平均影响分")
        fig.add_vline(x=0.85, line_dash="dot", annotation_text="高置信阈值")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("高危实体穿透 (Impact Score > 20)")
        st.table(df[df["impact_score"] > 20][["text", "type", "impact_score", "source_doc"]].sort_values("impact_score", ascending=False))
    else:
        st.info("暂无提取数据，请先在“文档分析”页进行处理。")

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
