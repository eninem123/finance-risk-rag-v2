# 贡献指南

感谢您考虑为 Finance-Risk-RAG 做出贡献！

## 如何贡献

### 报告问题

如果您发现了bug或有功能建议，请通过 [GitHub Issues](https://github.com/eninem123/finance-risk-rag-v2/issues) 提交。

提交问题时，请包含：
- 问题的清晰描述
- 复现步骤
- 期望行为与实际行为
- 环境信息（Python版本、操作系统等）
- 相关日志或截图

### 提交代码

1. **Fork 仓库**
   ```bash
   git clone https://github.com/eninem123/finance-risk-rag-v2.git
   cd finance-risk-rag-v2
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/your-bug-fix
   ```

3. **开发并测试**
   - 遵循代码规范
   - 添加必要的测试
   - 确保所有测试通过

4. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   ```

5. **推送并创建PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   然后在GitHub上创建Pull Request。

## 代码规范

### 提交信息规范

使用约定式提交格式：

- `feat:` 新功能
- `fix:` 修复bug
- `docs:` 文档更新
- `style:` 代码格式调整（不影响功能）
- `refactor:` 代码重构
- `perf:` 性能优化
- `test:` 测试相关
- `chore:` 构建过程或辅助工具的变动

示例：
```
feat: 添加风险趋势分析功能

- 实现时间序列风险计算
- 添加趋势预测算法
- 更新相关文档
```

### Python代码规范

- 遵循 PEP 8 规范
- 使用 Black 格式化代码（行宽100）
- 使用 isort 排序导入
- 使用 mypy 进行类型检查
- 所有函数和类都需要文档字符串

### 代码审查流程

1. 所有PR都需要至少一个审查者批准
2. CI检查必须通过
3. 代码覆盖率不应下降
4. 文档需要同步更新

## 开发环境设置

详见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

## 行为准则

- 尊重所有参与者
- 接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

## 许可证

通过贡献代码，您同意您的贡献将在 MIT 许可证下发布。
