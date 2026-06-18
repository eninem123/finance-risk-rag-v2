"""
Finance-Risk-RAG Streamlit 可视化仪表盘
======================================
"""

import sys
from pathlib import Path

# 添加 src 到路径以确保导入正常
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402
from finance_risk_rag.utils import load_json_file  # noqa: E402

st.set_page_config(
    page_title="Finance-Risk-RAG v2.2 | 财务风控分析",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def get_service():
    return RiskAnalysisService()


def main():
    st.sidebar.title("🏦 Finance-Risk-RAG v2.2")
    st.sidebar.markdown("---")

    menu = ["数据总览", "文档分析", "风险问答 (RAG)", "配置查看"]
    choice = st.sidebar.selectbox("功能菜单", menu)

    service = get_service()
    config = get_config()

    if choice == "数据总览":
        st.header("📊 财务风险监控大盘")

        # 加载分类结果
        class_data = load_json_file(config.docs_dir / "classification.json")
        if class_data:
            df = pd.DataFrame.from_dict(class_data, orient="index")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("文档类型分布")
                st.bar_chart(df["type"].value_counts())

            with col2:
                st.subheader("最近处理列表")
                st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无处理数据，请先在‘文档分析’中处理 PDF。")

    elif choice == "文档分析":
        st.header("📑 文档自动化识别与风险分析")

        pdf_files = list(config.docs_dir.glob("*.pdf"))
        if not pdf_files:
            st.warning(f"目录 {config.docs_dir} 中未找到 PDF 文件。")
            return

        selected_pdf = st.selectbox("选择要分析的文档", [f.name for f in pdf_files])

        if st.button("开始深度分析"):
            with st.spinner("正在执行 OCR、分类及风险实体提取..."):
                report = service.analyze_document(config.docs_dir / selected_pdf)

                st.success("分析完成！")

                # 展示分类信息
                st.subheader("📄 基本信息")
                c1, c2, c3 = st.columns(3)
                c1.metric("文档类型", report["classification"]["type"])
                c2.metric("置信度", f"{report['classification']['confidence']:.2%}")
                c3.metric("风险等级", report["risk_analysis"]["risk_level"])

                # 展示风险实体
                st.subheader("🔍 识别到的风险实体")
                entities = report["risk_analysis"]["entities"]
                if entities:
                    ent_df = pd.DataFrame(entities)
                    st.table(ent_df[["type", "text", "risk_score", "source"]])

                    st.subheader("💡 风险总结")
                    st.write(report["summary"])
                else:
                    st.info("该文档未检测到预定义的风险实体。")

    elif choice == "风险问答 (RAG)":
        st.header("🧠 风险智能咨询 (RAG)")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("询问有关财务风险的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    res = service.query_risk(prompt)
                    st.markdown(res["answer"])

                    if res["sources"]:
                        with st.expander("查看参考来源"):
                            for src in res["sources"]:
                                st.write(f"- {src.get('source')} (Chunk {src.get('chunk_index')})")

            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})

    elif choice == "配置查看":
        st.header("⚙️ 系统配置")
        st.json(
            {
                "LLM 模型": config.llm_model_name,
                "OCR DPI": config.ocr_dpi,
                "ChromaDB 路径": str(config.chroma_db_dir),
                "文档目录": str(config.docs_dir),
            }
        )


if __name__ == "__main__":
    main()
