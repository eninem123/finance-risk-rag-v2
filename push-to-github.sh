#!/bin/bash
# Finance-Risk-RAG GitHub推送脚本
# 使用方法: ./push-to-github.sh

set -e

echo "=========================================="
echo "  Finance-Risk-RAG GitHub推送脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}错误: 未找到Git，请先安装Git${NC}"
    exit 1
fi

# 获取当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}✓${NC} 当前目录: $SCRIPT_DIR"
echo ""

# 检查Git仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}初始化Git仓库...${NC}"
    git init
    git branch -m main
fi

# 检查远程仓库
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$REMOTE_URL" ]; then
    echo -e "${YELLOW}未配置远程仓库${NC}"
    echo ""
    echo "请选择操作:"
    echo "1) 推送到原仓库 (eninem123/finance-risk-rag-v2)"
    echo "2) 推送到自定义仓库"
    echo ""
    read -p "请输入选项 (1或2): " choice
    
    if [ "$choice" = "1" ]; then
        REMOTE_URL="https://github.com/eninem123/finance-risk-rag-v2.git"
        git remote add origin "$REMOTE_URL"
        echo -e "${GREEN}✓${NC} 已添加远程仓库: $REMOTE_URL"
    elif [ "$choice" = "2" ]; then
        read -p "请输入您的GitHub仓库URL: " REMOTE_URL
        git remote add origin "$REMOTE_URL"
        echo -e "${GREEN}✓${NC} 已添加远程仓库: $REMOTE_URL"
    else
        echo -e "${RED}无效选项${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} 远程仓库: $REMOTE_URL"
fi

echo ""

# 检查Git配置
if ! git config user.name &> /dev/null || ! git config user.email &> /dev/null; then
    echo -e "${YELLOW}配置Git用户信息...${NC}"
    
    if ! git config user.name &> /dev/null; then
        read -p "请输入Git用户名: " git_name
        git config user.name "$git_name"
    fi
    
    if ! git config user.email &> /dev/null; then
        read -p "请输入Git邮箱: " git_email
        git config user.email "$git_email"
    fi
    
    echo -e "${GREEN}✓${NC} Git配置完成"
fi

echo ""

# 显示提交历史
echo -e "${YELLOW}提交历史:${NC}"
git log --oneline -5
echo ""

# 推送到GitHub
echo -e "${YELLOW}推送到GitHub...${NC}"
echo ""

# 检查是否有未提交的更改
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}检测到未提交的更改，正在提交...${NC}"
    git add .
    git commit -m "chore: 自动提交未保存的更改"
    echo -e "${GREEN}✓${NC} 更改已提交"
    echo ""
fi

# 推送
echo "正在推送到 origin/main..."
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}  ✓ 推送成功!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo ""
    echo "仓库地址:"
    echo "  $REMOTE_URL"
    echo ""
    echo "后续操作:"
    echo "  1. 访问GitHub查看代码"
    echo "  2. 创建v2.0.0标签发布新版本"
    echo "  3. 在GitHub上启用Actions"
    echo ""
else
    echo ""
    echo -e "${RED}==========================================${NC}"
    echo -e "${RED}  ✗ 推送失败${NC}"
    echo -e "${RED}==========================================${NC}"
    echo ""
    echo "可能的解决方案:"
    echo "  1. 检查网络连接"
    echo "  2. 确认有仓库写入权限"
    echo "  3. 如果使用HTTPS，尝试使用SSH:"
    echo "     git remote set-url origin git@github.com:用户名/仓库名.git"
    echo "  4. 如果仓库已有内容，可能需要先拉取:"
    echo "     git pull origin main --rebase"
    echo ""
    exit 1
fi
