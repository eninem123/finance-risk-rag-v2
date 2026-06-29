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
st.sidebar.markdown("**银行级多语言财务文本风控系统**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "风险矩阵", "文档分析", "风险检索"],
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
        else:
            st.info("暂无处理数据。")
    else:
        st.info("请先执行文档处理。")

elif page == "风险矩阵":
    st.title("🎯 风险矩阵 (Risk Matrix)")
    st.markdown("基于 **影响程度 (Impact)** 与 **发生概率 (Confidence)** 的风险量化视图。")

    # 尝试加载已有的实体数据
    entities_file = config.docs_dir / "entities_extracted.json"
    if entities_file.exists():
        data = load_json_file(entities_file)
        if "entities" in data:
            df = pd.DataFrame(data["entities"])
            if not df.empty:
                # 绘制散点图作为风险矩阵
                import plotly.express as px

                fig = px.scatter(
                    df,
                    x="confidence",
                    y="impact_score",
                    size="risk_score",
                    color="risk_category",
                    hover_name="text",
                    labels={
                        "confidence": "置信度 (Probability)",
                        "impact_score": "影响程度 (Impact)",
                    },
                    title="风险实体分布矩阵",
                    range_x=[0, 1.1],
                    range_y=[0, 6],
                )
                # 添加象限背景线 (简化实现)
                fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=6, line=dict(color="Red", dash="dash"))
                fig.add_shape(type="line", x0=0, y0=3, x1=1.1, y1=3, line=dict(color="Red", dash="dash"))

                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("风险类别占比")
                    cat_counts = df["risk_category"].value_counts()
                    st.write(cat_counts)
                with col2:
                    st.subheader("高风险实体 Top 5")
                    st.dataframe(df.sort_values("risk_score", ascending=False).head(5)[["text", "risk_category", "risk_score"]])
            else:
                st.info("尚未提取到风险实体。")
    else:
        st.info("请先在'文档分析'页面提取风险实体。")

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

                    # 自动保存一份结果用于风险矩阵展示
                    from src.finance_risk_rag.utils import save_json_file
                    save_json_file(result.to_dict(), config.docs_dir / "entities_extracted.json")
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
