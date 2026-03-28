# Learn Claude Code OpenAI

这是基于https://learn.shareai.run/zh/ 项目的辅助开发,用于在中国大陆基于阿里百炼的千问模型API调用,来实现该项目的学习
改写了模型调用,模型返回内容的引用方法,适配penai规约
用独立可运行的 Python 脚本，从「最小 Agent 循环」一路演示到「任务、团队、自治与工作树隔离」——相当于把 **Claude Code / 类 Cursor 编码智能体** 的核心模式拆开、逐层叠加，便于阅读与实验。

## 适用场景

- 理解 **tool-use 循环**、工具分发、上下文与状态外置等常见设计。
- 对照源码做二次开发或教学，而非开箱即用的生产级产品。

## 环境要求

- **Python 3.10+**（脚本中使用了 `Path | None` 等类型标注）。
- 兼容 **OpenAI Chat Completions** 的 HTTP API（默认配置为阿里云 DashScope 兼容端点）。
- 部分脚本需要 **Git**（如 `s12_worktree_task_isolation.py` 中的 worktree）。

## 安装

```bash
cd learn-claude-code-openai   # 若本地目录名不同，请改为你的仓库根目录
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置（`.env`）

在项目目录或当前工作目录放置 `.env`，常用变量如下：

| 变量 | 说明 |
|------|------|
| `llm_api_key` | API Key；未设置时，客户端可能回退读取 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`（以各脚本注释为准） |
| `llm_base_url` | API 基地址；默认多为 DashScope 兼容模式地址 |
| `llm_model` | 模型名；默认多为 `qwen3.5-plus` |

按需修改为你的服务商与模型。

## 运行方式

每个 `sNN_*.py` 都是可直接执行的演示入口，在**目标工作区目录**下运行（脚本会以当前工作目录为「工作空间」）：

```bash
python s01_agent_loop.py
```

后续模块在 `s01` 的基础上增加工具、持久化、多线程等能力；建议按编号顺序阅读源码顶部的模块说明（docstring）。

## 模块一览

| 文件 | 主题 |
|------|------|
| `s01_agent_loop.py` | **Agent 循环**：`while` 中调用 LLM → 执行 tool → 将结果写回 messages，直到不再请求工具。 |
| `s02_tool_use.py` | **多工具**：在循环不变的前提下扩展 `TOOLS` 与分发逻辑（如读/写/编辑）。 |
| `s03_todo_write.py` | **TodoWrite**：模型通过工具维护任务列表，并带「久未更新则提醒」机制。 |
| `s04_subagent.py` | **子智能体**：子会话独立上下文，结束后只把摘要返回父会话。 |
| `s05_skill_loading.py` | **Skills**：系统提示里只放技能元数据，`load_skill` 再按需注入全文。 |
| `s06_context_compact.py` | **上下文压缩**：微压缩、自动摘要、手动 `compact` 等多层策略。 |
| `s07_task_system.py` | **任务系统**：`.tasks/task_*.json` 持久化任务与依赖图。 |
| `s08_background_tasks.py` | **后台任务**：线程执行长命令，主循环在调用 LLM 前拉取完成通知。 |
| `s09_agent_teams.py` | **团队**：命名成员、各自 agent 循环、基于 JSONL 收件箱通信。 |
| `s10_team_protocols.py` | **团队协议**：关闭流程、计划审批等，基于 `request_id` 关联请求与响应。 |
| `s11_autonomous_agents.py` | **自治代理**：空闲轮询任务板、认领任务、压缩后身份再注入等。 |
| `s12_worktree_task_isolation.py` | **Worktree + 任务隔离**：目录级隔离、事件总线、与任务 ID 协同。 |

## 仓库内其他内容

- `skills/`：供 `s05` 演示用的技能目录示例（如 `skills/pdf/SKILL.md`）。

## 许可证

见仓库根目录 [LICENSE](LICENSE)（MIT）。
