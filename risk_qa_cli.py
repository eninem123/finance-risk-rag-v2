# risk_qa_cli.py
import os
import argparse
from rag_core import query, build_db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="重新构建向量库（从 docs/*.txt）")
    parser.add_argument("--ask", type=str, help="向系统提问")
    args = parser.parse_args()
    if args.build:
        print("开始构建向量库...")
        build_db()
        print("构建完成。")
    if args.ask:
        print("查询中...")
        res = query(args.ask)
        if res is None:
            print("查询或回答失败，请检查日志。")
        else:
            print("回答：")
            print(res["answer"])
            print("\n相关片段元信息：")
            for s in res.get("sources", []):
                print(s)

if __name__ == "__main__":
    main()


#测试
# 构建 DB
#python risk_qa_cli.py --build

# 提问
#python risk_qa_cli.py --ask "这份报告的主要风险是什么？"
