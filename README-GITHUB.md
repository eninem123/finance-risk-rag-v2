# 推送到GitHub操作指南

## 快速推送

### 方法一：使用自动脚本（推荐）

```bash
# 1. 进入项目目录
cd finance-risk-rag-optimized

# 2. 运行推送脚本
./push-to-github.sh
```

脚本会引导您完成：
- 配置Git用户信息
- 添加远程仓库
- 推送到GitHub

### 方法二：手动推送

```bash
# 1. 进入项目目录
cd finance-risk-rag-optimized

# 2. 添加远程仓库
git remote add origin https://github.com/eninem123/finance-risk-rag-v2.git

# 3. 推送到GitHub
git push -u origin main
```

## 推送后操作

### 1. 创建v2.0.0标签

```bash
# 创建标签
git tag -a v2.0.0 -m "发布v2.0.0 - 全面代码优化"

# 推送标签到GitHub
git push origin v2.0.0
```

### 2. 在GitHub上启用Actions

1. 访问仓库页面
2. 点击 "Actions" 标签
3. 点击 "I understand my workflows, go ahead and enable them"

### 3. 设置分支保护（可选）

1. 访问 Settings > Branches
2. 点击 "Add rule"
3. 选择 "main" 分支
4. 启用以下选项：
   - Require pull request reviews before merging
   - Require status checks to pass before merging
   - Require branches to be up to date before merging

## 文件清单

已创建并提交的文件：

```
finance-risk-rag-optimized/
├── .github/
│   └── workflows/
│       ├── ci.yml          # CI工作流
│       └── release.yml     # 发布工作流
├── docs/
│   ├── API.md              # API文档
│   └── DEVELOPMENT.md      # 开发指南
├── .env.example            # 环境变量示例
├── .gitignore              # Git忽略规则
├── CHANGELOG.md            # 变更日志
├── CONTRIBUTING.md         # 贡献指南
├── LICENSE                 # MIT许可证
├── OPTIMIZATION_REPORT.md  # 优化报告
├── README.md               # 项目文档
├── config.py               # 配置模块
├── extract_entities.py     # 实体提取模块
├── rag_core.py             # RAG核心模块
├── requirements.txt        # 依赖清单
├── utils.py                # 工具模块
└── push-to-github.sh       # 推送脚本
```

## 常见问题

### Q: 推送时出现权限错误？

A: 请检查：
1. 是否有仓库的写入权限
2. 是否配置了正确的GitHub凭据
3. 尝试使用SSH方式：`git remote set-url origin git@github.com:eninem123/finance-risk-rag-v2.git`

### Q: 仓库已有内容如何处理？

A: 如果原仓库已有内容，建议：
```bash
# 先拉取原仓库内容
git pull origin main --rebase

# 解决冲突后推送
git push -u origin main
```

### Q: 如何验证推送成功？

A: 访问：https://github.com/eninem123/finance-risk-rag-v2

## 联系支持

如有问题，请在GitHub上创建Issue。
