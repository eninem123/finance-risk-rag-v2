import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

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


@st.cache_data
def get_cached_extraction(file_path):
    return service.pipeline.process(file_path)


st.sidebar.title("🏦 Finance-Risk-RAG v2.3")
st.sidebar.markdown("银行级多语言财务文本风控系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "文档分析", "风险矩阵", "风险检索", "风险报告"],
)

if page == "数据总览":
    st.title("📊 数据总览")

    # Load classification data
    class_path = config.docs_dir / "classification.json"
    if class_path.exists():
        data = load_json_file(class_path)
        if data:
            df = pd.DataFrame.from_dict(data, orient="index")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("文档类型分布")
                type_counts = df["type"].value_counts().reset_index()
                type_counts.columns = ["类型", "数量"]
                fig = px.pie(type_counts, values="数量", names="类型", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("处理详情列表")
                st.dataframe(
                    df[["type", "confidence"]].rename(
                        columns={"type": "类别", "confidence": "置信度"}
                    ),
                    use_container_width=True,
                )
        else:
            st.info("暂无处理数据。")
    else:
        st.info("请先在 CLI 执行 `python main.py process`。")

elif page == "文档分析":
    st.title("📑 文档分析")

    txt_files = list(config.docs_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

    if not txt_files:
        st.warning("未发现处理后的文档文本。")
    else:
        selected_file = st.selectbox("选择文档", [f.name for f in txt_files])
        txt_path = config.docs_dir / selected_file

        if st.button("执行深度风险提取"):
            with st.spinner("正在进行多引擎提取..."):
                result = get_cached_extraction(txt_path)

                c1, c2, c3 = st.columns(3)
                c1.metric("风险等级", result.risk_level)
                c2.metric("综合评分", result.total_risk_score)
                c3.metric("检测到实体", len(result.entities))

                if result.entities:
                    st.subheader("风险实体明细")
                    entities_df = pd.DataFrame([e.to_dict() for e in result.entities])
                    st.dataframe(
                        entities_df[["type", "text", "risk_score", "confidence", "source"]].rename(
                            columns={
                                "type": "类型",
                                "text": "内容",
                                "risk_score": "风险分",
                                "confidence": "置信度",
                                "source": "来源",
                            }
                        ),
                        use_container_width=True,
                    )
                else:
                    st.success("未检测到显著风险实体。")

elif page == "风险矩阵":
    st.title("🛡️ 风险矩阵 (Risk Matrix)")
    st.markdown("可视化展示风险影响程度与识别置信度。")

    txt_files = list(config.docs_dir.glob("*.txt"))
    txt_files = [f for f in txt_files if f.name != "all_extracted.txt"]

    all_entities = []
    for f in txt_files:
        res = get_cached_extraction(f)
        for e in res.entities:
            d = e.to_dict()
            d["filename"] = f.stem
            all_entities.append(d)

    if all_entities:
        df = pd.DataFrame(all_entities)
        fig = px.scatter(
            df,
            x="confidence",
            y="risk_score",
            color="type",
            size="risk_score",
            hover_data=["text", "filename"],
            labels={"confidence": "提取置信度", "risk_score": "风险影响评分"},
            title="风险分布：影响 vs 置信度",
        )
        # Add quadrants
        fig.add_hline(y=15, line_dash="dot", annotation_text="高影响临界")
        fig.add_vline(x=0.8, line_dash="dot", annotation_text="高置信阈值")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无足够数据构建矩阵。")

elif page == "风险检索":
    st.title("🔍 智能风控 RAG")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("询问关于已处理文档的风险细节..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索知识库并生成回答..."):
                try:
                    res = service.query_risk(prompt)
                    response = res.answer
                    st.markdown(response)

                    if res.sources:
                        with st.expander("数据溯源 (Reference Chunks)"):
                            for src in res.sources:
                                st.write(
                                    f"📄 **{src.get('source')}** (块: {src.get('chunk_index')})"
                                )
                except Exception as e:
                    response = f"检索失败: {e}"
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "风险报告":
    st.title("📋 专业风险报告生成")

    pdf_files = list(config.docs_dir.glob("*.pdf"))
    if not pdf_files:
        st.warning("docs 目录下未发现 PDF 文件。")
    else:
        selected_pdf = st.selectbox("选择目标 PDF", [f.name for f in pdf_files])
        pdf_path = config.docs_dir / selected_pdf

        if st.button("生成 & 预览报告"):
            with st.spinner("正在生成银行级分析报告..."):
                analysis = service.analyze_document(pdf_path)
                report_md = service.generate_report(analysis)

                st.markdown("---")
                st.markdown(report_md)

                st.download_button(
                    label="📥 下载 Markdown 报告",
                    data=report_md,
                    file_name=f"Risk_Report_{pdf_path.stem}.md",
                    mime="text/markdown",
                )
