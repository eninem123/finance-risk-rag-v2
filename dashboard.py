import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
import streamlit as st

from finance_risk_rag.config import get_config  # noqa: E402
from finance_risk_rag.service import RiskAnalysisService  # noqa: E402

st.set_page_config(page_title="Finance-Risk-RAG Dashboard", layout="wide")


def main():
    st.title("🏦 Finance-Risk-RAG v2.2 Dashboard")
    st.sidebar.header("控制面板")

    config = get_config()
    service = RiskAnalysisService(config)

    menu = ["概览", "文档分析", "风险实体挖掘", "RAG 问答"]
    choice = st.sidebar.selectbox("功能选择", menu)

    if choice == "概览":
        st.subheader("系统概览")
        st.write("欢迎使用银行级财务文本风控 AI 系统。")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("支持实体类", "17+")
        with col2:
            st.metric("OCR 分辨率", f"{config.ocr_dpi} DPI")
        with col3:
            st.metric("系统版本", "v2.2")

    elif choice == "文档分析":
        st.subheader("文档分类与 OCR 统计")
        if st.button("运行批量分析"):
            with st.spinner("正在分析文档..."):
                reports = service.generate_batch_report()
                if reports:
                    df = pd.DataFrame(
                        [
                            {
                                "文件名": r["filename"],
                                "分类": r["classification"]["type"],
                                "置信度": r["classification"]["confidence"],
                                "风险评分": r["extraction"]["total_risk_score"],
                                "风险等级": r["extraction"]["risk_level"],
                            }
                            for r in reports
                        ]
                    )
                    st.table(df)
                else:
                    st.warning("未找到待处理文档。")

    elif choice == "风险实体挖掘":
        st.subheader("风险实体提取详情")
        pdf_files = list(config.docs_dir.glob("*.pdf"))
        selected_pdf = st.selectbox("选择文档", [f.name for f in pdf_files])

        if st.button("提取实体") and selected_pdf:
            pdf_path = config.docs_dir / selected_pdf
            with st.spinner("提取中..."):
                res = service.run_full_analysis(pdf_path)
                st.write(f"### 风险等级: {res['extraction']['risk_level']}")

                entities = res["extraction"]["entities"]
                if entities:
                    ent_df = pd.DataFrame(entities)
                    st.dataframe(ent_df[["type", "text", "risk_score", "confidence", "source"]])
                else:
                    st.info("未发现明显风险实体。")

    elif choice == "RAG 问答":
        st.subheader("风险咨询 (RAG)")
        query = st.text_input(
            "请输入您的问题：", placeholder="例如：这几份报告中提到的共同风险有哪些？"
        )
        if query:
            with st.spinner("思考中..."):
                result = service.engine.query(query)
                st.markdown(f"**回答：**\n\n{result.answer}")
                with st.expander("查看来源"):
                    st.json(result.sources)


if __name__ == "__main__":
    main()
