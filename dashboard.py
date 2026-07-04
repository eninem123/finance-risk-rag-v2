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


def get_risk_color(level: str) -> str:
    colors = {"低风险": "green", "中风险": "orange", "高风险": "red", "极高风险": "darkred"}
    return colors.get(level, "gray")


st.sidebar.title("🏦 Finance-Risk-RAG v2.3")
st.sidebar.markdown("银行级多语言财务文本风控系统")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "功能菜单",
    ["数据总览", "文档分析", "风险检索"],
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
                fig_type = px.pie(df, names="type", hole=0.4, title="文档类型占比")
                st.plotly_chart(fig_type, use_container_width=True)

            with col2:
                st.subheader("处理详情")
                st.dataframe(df[["type", "confidence"]], use_container_width=True)

            # Risk Matrix (Mock data for illustration if not processed yet)
            st.markdown("---")
            st.subheader("🎯 风险矩阵 (Risk Matrix)")
            # Assuming we have some processed results in log for matrix
            log_path = config.processing_log_path
            log_data = load_json_file(log_path)
            if log_data:
                matrix_data = []
                for name, info in log_data.items():
                    # We might not have full risk info here yet if only processed
                    # This is just for demonstration of the v2.3 visual capability
                    matrix_data.append(
                        {
                            "name": name,
                            "type": info.get("classification", {}).get("type", "未知"),
                            "confidence": info.get("classification", {}).get("confidence", 0.5),
                            "risk_score": 0,  # Placeholder
                        }
                    )
                # If we had full analysis results, we'd use them here
                st.info("风险矩阵需要运行 '文档分析' 以获取完整量化分值。")
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

                col1, col2, col3 = st.columns(3)
                col1.metric("风险等级", result.risk_level)
                col2.metric("总风险评分", result.total_risk_score)
                col3.metric("识别实体数", len(result.entities))

                if result.entities:
                    entities_df = pd.DataFrame([e.to_dict() for e in result.entities])

                    st.subheader("实体分布")
                    fig_dist = px.bar(entities_df, x="type", color="type", title="各类型风险实体统计")
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.subheader("详细清单")
                    st.dataframe(entities_df, use_container_width=True)

                    # Export report
                    if st.button("生成详细报告"):
                        # Dummy call for demo
                        st.success("报告生成逻辑已就绪。")
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
