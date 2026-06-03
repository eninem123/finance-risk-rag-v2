"""
Finance-Risk-RAG 风险问答命令行工具
==================================

提供交互式界面，支持构建向量索引和执行智能风控问答。

作者: Finance-Risk-RAG Team
版本: 2.0.0
"""

import argparse

from config import get_config
from rag_core import RAGEngine


def run_interactive(engine: RAGEngine, top_k: int) -> None:
    """运行交互式模式"""
    print("\n" + "*" * 50)
    print("  Finance-Risk-RAG 交互式问答系统 (输入 'exit' 退出)")
    print("*" * 50)

    while True:
        try:
            question = input("\n[问]: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "退出", "q"]:
                print("再见！")
                break

            result = engine.query(question, top_k=top_k)
            print(f"\n[答]: {result.answer}")

        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n[错误]: {e}")


def main() -> None:
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="Finance-Risk-RAG 风险问答系统")
    parser.add_argument(
        "--build", action="store_true", help="构建/更新向量数据库（基于 docs/*.txt）"
    )
    parser.add_argument("--ask", type=str, help="执行一次性查询")
    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互式对话模式")
    parser.add_argument("--top-k", type=int, default=4, help="检索的相关文档块数量 (默认: 4)")

    args = parser.parse_args()

    config = get_config()
    engine = RAGEngine(docs_dir=str(config.docs_dir), db_path=str(config.chroma_db_dir))

    if args.build:
        print("正在构建向量数据库，请稍候...")
        stats = engine.build_index()
        print(f"构建完成！统计信息: {stats}")

    if args.ask:
        print(f"查询中: {args.ask}")
        result = engine.query(args.ask, top_k=args.top_k)
        print("\n" + "=" * 50)
        print("回答：")
        print(result.answer)
        print("\n相关来源：")
        for i, source in enumerate(result.sources, 1):
            print(
                f"  [{i}] {source.get('source', '未知')} (分块: {source.get('chunk_index', 'N/A')})"
            )
        print("=" * 50)

    if args.interactive:
        run_interactive(engine, args.top_k)

    if not any([args.build, args.ask, args.interactive]):
        parser.print_help()


if __name__ == "__main__":
    main()
