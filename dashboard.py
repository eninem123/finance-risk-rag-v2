import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# 添加 src 到路径
sys.path.append(str(Path(__file__).parent / "src"))

from finance_risk_rag.config import get_config
from finance_risk_rag.service import RiskAnalysisService

st.set_page_config(page_title="Finance-Risk-RAG Dashboard", layout="wide")

def main():
    st.title("🏦 Finance-Risk-RAG 智能风控仪表盘")
    st.sidebar.title("控制面板")

    config = get_config()
    service = RiskAnalysisService(config)

    menu = ["项目概览", "单文档深度分析", "RAG 智能问答"]
    choice = st.sidebar.selectbox("切换功能", menu)

    if choice == "项目概览":
        st.subheader("📊 系统状态与处理历史")
        # 这里可以读取 processing_log.json 展示历史
        st.info("欢迎使用银行级多语言财务文本风控 AI 系统。请在左侧选择功能。")

    elif choice == "单文档深度分析":
        st.subheader("🔍 文档风险扫描")
        uploaded_file = st.file_uploader("上传财务 PDF 文档", type=["pdf"])

        if uploaded_file is not None:
            # 保存临时文件
            temp_path = Path("cache") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("正在进行深度 AI 分析..."):
                try:
                    report = service.analyze_document(temp_path)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("文档类型", report["document"]["type"])
                    col2.metric("风险等级", report["risk_analysis"]["risk_level"])
                    col3.metric("风险得分", report["risk_analysis"]["total_risk_score"])

                    st.write(f"**AI 摘要:** {report['summary']}")

                    # 展示实体
                    entities = report["risk_analysis"]["entities"]
                    if entities:
                        st.write("### 🚩 识别到的风险点")
                        df = pd.DataFrame(entities)
                        st.dataframe(df[["type", "text", "risk_score", "confidence", "source"]], use_container_width=True)
                    else:
                        st.success("未发现预定义风险点。")
                except Exception as e:
                    st.error(f"分析失败: {e}")

    elif choice == "RAG 智能问答":
        st.subheader("💬 风险知识库问答")

        if st.button("构建/更新向量索引"):
            with st.spinner("正在索引文档..."):
                service.rag_engine.build_index()
                st.success("索引构建完成！")

        query = st.text_input("请输入您关于财务风险的问题：")
        if query:
            with st.spinner("正在检索分析..."):
                result = service.run_query(query)
                st.write(f"**回答:** {result.answer}")
                if result.sources:
                    with st.expander("查看参考来源"):
                        st.json(result.sources)

if __name__ == "__main__":
    main()
