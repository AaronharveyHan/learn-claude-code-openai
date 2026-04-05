# Learn Claude Code OpenAI

用独立可运行的 Python 脚本，从「最小 Agent 循环」一路演示到「MCP 工具对接、任务系统、多智能体团队与 Worktree 隔离」——把 **Claude Code / 类 Cursor 编码智能体** 的核心模式拆开、逐层叠加，便于阅读与实验。

项目基于 [learn.shareai.run](https://learn.shareai.run/zh/)，改写了模型调用方式，适配 **OpenAI 兼容规约**，默认对接阿里云 DashScope（千问模型），也可替换为任意兼容 API。

## 适用场景

- 理解 tool-use 循环、工具分发、上下文压缩、状态外置等常见设计模式
- 了解多智能体团队、任务系统、MCP 协议等进阶架构
- 对照源码做二次开发或教学，而非开箱即用的生产级产品

## 环境要求

- **Python 3.10+**（使用了 `X | Y` 类型标注语法）
- 兼容 **OpenAI Chat Completions** 规约的 HTTP API
- 部分脚本需要本地安装 **Git**（`s12` 使用 worktree）

## 安装

```bash
git clone <repo-url>
cd Claude_Code
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` 包含核心依赖与可选依赖：

| 包 | 说明 |
|----|------|
| `openai` | 必须，所有脚本均使用 |
| `python-dotenv` | 必须，读取 `.env` 配置 |
| `tiktoken` | 可选，`s06` token 计数更精准（不安装时自动降级到字符估算） |
| `mcp` | 可选，`s13` MCP 协议对接（不安装时 s13 无法运行） |

## 配置

复制 `.env.example` 为 `.env` 并填写你的 API 信息：

```bash
cp .env.example .env
```

| 变量 | 必填 | 说明 | 示例 |
|------|------|------|------|
| `llm_api_key` | ✅ | API 密钥 | `sk-xxx` |
| `llm_base_url` | 可选 | API 地址（使用 OpenAI 官方 API 时可省略） | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `llm_model` | 可选 | 模型名称 | `qwen-plus` |
| `llm_context_window` | 可选 | 上下文窗口大小，默认 `131072`（s06） | `65536` |
| `llm_max_tokens` | 可选 | 单次生成上限，默认 `8000`（s06） | `4096` |
| `llm_estimate_margin` | 可选 | 压缩阈值缓冲系数，默认 `0.8`（s06） | `0.75` |

## 运行

每个 `sNN_*.py` 都是可直接执行的独立演示，在目标工作目录下运行（脚本以当前目录为工作空间）：

```bash
python s01_agent_loop.py
python s13_mcp_tools.py "列出项目文件并简要介绍"
```

建议按编号顺序阅读，每个脚本顶部的 docstring 都有该模块的设计说明。

## 模块一览

| 文件 | 主题 | 核心模式 |
|------|------|----------|
| `s01_agent_loop.py` | **Agent 循环** | `while tool_use` 循环：LLM → 执行工具 → 结果写回 messages |
| `s02_tool_use.py` | **多工具** | Tool dispatch map，safe_path 安全路径校验 |
| `s03_todo_write.py` | **TodoWrite** | 模型通过工具维护任务列表，久未更新自动提醒 |
| `s04_subagent.py` | **子智能体** | 子会话独立上下文，只把摘要返回父会话 |
| `s05_skill_loading.py` | **Skills** | 系统提示仅放元数据，`load_skill` 按需注入全文（两层加载） |
| `s06_context_compact.py` | **上下文压缩** | 微压缩 → 自动摘要 → 手动 compact，tiktoken 精准计数 |
| `s07_task_system.py` | **任务系统** | `.tasks/task_*.json` 持久化任务与依赖图 |
| `s08_background_tasks.py` | **后台任务** | 线程执行长命令，主循环拉取完成通知 |
| `s09_agent_teams.py` | **团队** | 命名成员各自运行 agent 循环，JSONL 收件箱通信 |
| `s10_team_protocols.py` | **团队协议** | 关闭流程、计划审批，`request_id` 关联请求与响应 |
| `s11_autonomous_agents.py` | **自治代理** | 空闲轮询任务板、认领任务、压缩后身份再注入 |
| `s12_worktree_task_isolation.py` | **Worktree 隔离** | Git worktree 目录级隔离，事件总线，与任务 ID 协同 |
| `s13_mcp_tools.py` | **MCP 工具对接** | MCP schema → OpenAI 格式转换，stdio 子进程工具服务器 |
| `s14_streaming.py` | **流式输出** | `stream=True`，逐 token 实时打印；tool_call delta 累积模式 |

## 架构演进

```
s01  基础循环
 └─ s02  多工具 + 安全路径
     └─ s03  任务状态（内存）
         └─ s04  子智能体（上下文隔离）
             └─ s05  技能懒加载（提示词管理）
                 └─ s06  上下文压缩（token 预算）
                     └─ s07  任务持久化（文件系统）
                         └─ s08  后台并发（线程）
                             └─ s09  多智能体团队（JSONL 消息总线）
                                 └─ s10  团队治理协议
                                     └─ s11  自治 + 任务认领
                                         └─ s12  Worktree 执行隔离
                                             └─ s13  MCP 协议对接
                                                └─ s14  流式输出
```

**核心设计原则：循环不变，功能叠加。** s01 的 `while tool_use` 循环在 s13 中原封不动，所有能力都在外围叠加，不破坏核心结构。

## 仓库结构

```
.
├── s01_agent_loop.py  ─┐
├── ...                  ├─ 按序阅读的演示脚本
├── s14_streaming.py   ─┘
├── skills/
│   └── pdf/SKILL.md       # s05 技能示例
├── tests/                 # pytest 单元测试（66 用例）
├── .env.example            # 环境变量模板
├── requirements.txt
└── README.md
```

运行时产生的目录（已加入 `.gitignore`）：

| 目录 | 用途 |
|------|------|
| `.tasks/` | 任务 JSON 文件（s07+） |
| `.team/` | 团队配置与 JSONL 收件箱（s09+） |
| `.transcripts/` | 压缩后的对话记录（s06） |
| `.worktrees/` | Worktree 索引与事件日志（s12） |

## 单元测试

```bash
pip install pytest
pytest tests/ -v
```

覆盖范围（**66 个用例，全部通过**）：

| 测试文件 | 覆盖对象 | 用例数 |
|----------|----------|--------|
| `test_safe_path.py` | 路径安全校验：合法路径、`../` 穿越、绝对路径、符号链接逃逸 | 9 |
| `test_run_bash.py` | 命令执行：正常输出、危险命令拦截（5 种）、截断、stderr、已知绕过盲区 | 13 |
| `test_atomic_write.py` | 原子写入：正确性、覆盖、Unicode、失败回滚、无残留 .tmp | 10 |
| `test_message_bus.py` | 消息总线：收发、FIFO、清空、隔离、broadcast、并发安全、损坏行容错 | 14 |
| `test_task_manager.py` | 任务系统：CRUD、状态流转、双向依赖、完成清除依赖、并发 ID 无重复 | 20 |

> 测试不依赖真实 LLM API，使用假 key 注入，可在无网络环境运行。

## 工程质量说明

项目经过多轮审计与加固，当前综合评分 **9.0/10**。

### 已修复问题一览

| 类别 | 修复内容 | 涉及文件 |
|------|----------|----------|
| **原子文件写入** | `run_write` / `run_edit` / 任务存储 全部改为 `mkstemp + os.replace`，崩溃时不产生残缺文件 | s02–s12（全覆盖） |
| **日志权限** | `agent_debug.log` 创建后立即 `chmod 0o600`，防止同机用户读取 LLM 响应内容 | s01–s14（全部） |
| **并发安全** | MessageBus 为每个 inbox 独立加锁；TaskManager.create() 加 `threading.Lock` 防止 ID 重复 | s09、s07 |
| **线程退出** | 后台线程改为非 daemon；注册 `atexit.register(BG.join_all)` 兜底；idle 轮询改用 `Event.wait()` 即时响应信号 | s08–s11 |
| **状态持久化** | shutdown / plan 请求追踪写入 JSON 文件，重启后状态不丢失；加载时自动清理超过 1 小时的已完结记录 | s10 |
| **配置可变性** | 上下文压缩阈值（context window / max tokens / margin）改为从环境变量读取，无需修改代码 | s06 |
| **压缩后身份** | `auto_compact()` 从运行时 `messages` 提取实际 system message，而非硬编码全局常量 | s06 |
| **路径安全** | SKILL.md 加载前校验 `is_relative_to(skills_dir)`，阻止 symlink 指向工作目录外 | s05 |
| **MCP 权限** | 临时服务脚本写入后立即 `chmod 0o700`，防止同机用户替换或执行 | s13 |
| **流式累积** | `tool_calls` delta 中 `name` 字段改为赋值（`=`），防止多 chunk 重复拼接 | s14 |
| **损坏容错** | JSONL 收件箱含损坏行时跳过并记录 warning，不影响有效消息读取 | s09 |
| **Token 计数** | 优先使用 tiktoken 精准计数（中文比 `//4` 精确约 3 倍），未安装时自动降级 | s06 |
| **UUID 完整性** | plan_approval / shutdown 的 request_id 使用完整 36 位 UUID，消除高并发碰撞 | s10 |

### 已知限制（教学项目范围内）

| 项目 | 说明 |
|------|------|
| 危险命令检测 | 基于字符串子串匹配，`rm -r -f`（拆分标志）等写法不被拦截（已有测试记录该盲区） |

> 这是教学项目，不建议直接用于生产环境。

## 许可证

[MIT](LICENSE)
