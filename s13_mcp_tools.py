#!/usr/bin/env python3
"""
s13_mcp_tools.py - MCP Tool Integration

Model Context Protocol (MCP) is the open standard that Claude Code uses
to connect to external tool servers. This script shows the full bridge:

    +----------+    stdio / JSON-RPC    +--------------------+
    |  Agent   | <-------------------> |    MCP Server      |
    | (OpenAI) |   1. list_tools()     | run_bash / read /  |
    +----------+   2. call_tool()      | write / edit       |
                                       +--------------------+

Key patterns:
    1. Discovery  — list_tools() → convert MCP schema → OpenAI tool format
    2. Dispatch   — LLM tool_call → call_tool(name, args) → tool result
    3. Lifecycle  — spawn server subprocess → connect → agent loop → cleanup

The agent loop itself is unchanged from s01. All that changes is where
the tool schemas come from (MCP server instead of hand-written dicts)
and where tool calls are executed (MCP server instead of local functions).

Architecture:

    this process                         child process
    ────────────                         ─────────────
    ClientSession  ←── stdin/stdout ──→  FastMCP server
         │                                     │
    list_tools()                          @mcp.tool() bash/read/write/edit
    call_tool()
         │
    OpenAI agent loop

Run:
    pip install mcp
    python s13_mcp_tools.py "list files and show first 20 lines of s01"
"""
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Logger setup ──────────────────────────────────────────────────────────────
LOG_FILE = "agent_debug.log"

class ColorFormatter(logging.Formatter):
    """终端彩色输出，文件输出保持纯文本。"""
    COLORS = {
        logging.DEBUG:    "\033[90m",   # 灰
        logging.INFO:     "\033[36m",   # 青
        logging.WARNING:  "\033[33m",   # 黄
        logging.ERROR:    "\033[31m",   # 红
        logging.CRITICAL: "\033[35m",   # 紫
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)

def setup_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter(fmt, datefmt))
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    os.chmod(LOG_FILE, 0o600)   # 仅 owner 可读写，防止 LLM 响应内容被同机其他用户读取
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = setup_logger()

# ── Client / constants ────────────────────────────────────────────────────────
WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.getenv("llm_api_key"),
    base_url=os.getenv("llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("llm_model", "qwen-plus")
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

# ── Embedded MCP server script ────────────────────────────────────────────────
# 这段代码会被写入临时文件，作为子进程启动。
# FastMCP 会自动把带 @mcp.tool() 装饰的函数暴露为 MCP 工具。
MCP_SERVER_SCRIPT = (
    "import os\n"
    "import subprocess\n"
    "from pathlib import Path\n"
    "from mcp.server.fastmcp import FastMCP\n"
    "\n"
    "mcp = FastMCP('coding-agent-tools')\n"
    "WORKDIR = Path(os.getcwd())\n"
    "\n"
    "def _safe_path(path):\n"
    "    p = (WORKDIR / path).resolve()\n"
    "    if not str(p).startswith(str(WORKDIR.resolve())):\n"
    "        raise ValueError(f'Path outside workspace: {path!r}')\n"
    "    return p\n"
    "\n"
    "@mcp.tool()\n"
    "def run_bash(command: str) -> str:\n"
    "    # Run a shell command in the workspace directory.\n"
    "    dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', '> /dev/']\n"
    "    if any(d in command for d in dangerous):\n"
    "        return 'Error: Dangerous command blocked'\n"
    "    try:\n"
    "        r = subprocess.run(command, shell=True, cwd=WORKDIR,\n"
    "                           capture_output=True, text=True, timeout=120)\n"
    "        out = (r.stdout + r.stderr).strip()\n"
    "        return out[:50000] if out else '(no output)'\n"
    "    except subprocess.TimeoutExpired:\n"
    "        return 'Error: Timeout (120s)'\n"
    "    except Exception as e:\n"
    "        return f'Error: {e}'\n"
    "\n"
    "@mcp.tool()\n"
    "def read_file(path: str, limit: int = 200) -> str:\n"
    "    # Read a file from the workspace. limit = max lines to return.\n"
    "    try:\n"
    "        p = _safe_path(path)\n"
    "        lines = p.read_text(errors='replace').splitlines()\n"
    "        total = len(lines)\n"
    "        if limit and limit < total:\n"
    "            lines = lines[:limit] + [f'... ({total - limit} more lines)']\n"
    "        return '\\n'.join(lines)[:50000]\n"
    "    except Exception as e:\n"
    "        return f'Error: {e}'\n"
    "\n"
    "@mcp.tool()\n"
    "def write_file(path: str, content: str) -> str:\n"
    "    # Write content to a file in the workspace (creates parent dirs).\n"
    "    try:\n"
    "        p = _safe_path(path)\n"
    "        p.parent.mkdir(parents=True, exist_ok=True)\n"
    "        p.write_text(content)\n"
    "        return f'Wrote {len(content)} bytes to {path}'\n"
    "    except Exception as e:\n"
    "        return f'Error: {e}'\n"
    "\n"
    "@mcp.tool()\n"
    "def edit_file(path: str, old_text: str, new_text: str) -> str:\n"
    "    # Replace the first occurrence of old_text with new_text in a file.\n"
    "    try:\n"
    "        p = _safe_path(path)\n"
    "        original = p.read_text()\n"
    "        if old_text not in original:\n"
    "            return f'Error: Text not found in {path}'\n"
    "        p.write_text(original.replace(old_text, new_text, 1))\n"
    "        return f'Edited {path}'\n"
    "    except Exception as e:\n"
    "        return f'Error: {e}'\n"
    "\n"
    "if __name__ == '__main__':\n"
    "    mcp.run(transport='stdio')\n"
)

# ── MCP ↔ OpenAI 协议桥 ────────────────────────────────────────────────────────

async def mcp_list_tools(session) -> list[dict]:
    """
    从 MCP server 获取工具列表，转换为 OpenAI tool format。

    MCP tool schema:
        tool.name        → function.name
        tool.description → function.description
        tool.inputSchema → function.parameters  (已经是 JSON Schema)
    """
    result = await session.list_tools()
    tools = []
    for tool in result.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        })
    log.info("MCP   discovered %d tools: %s",
             len(tools), [t["function"]["name"] for t in tools])
    return tools


async def mcp_call_tool(session, name: str, args: dict) -> str:
    """
    通过 MCP session 调用工具，返回文本结果。

    MCP 返回的 result.content 是 ContentBlock 列表，
    每个 block 可能是 TextContent / ImageContent / EmbeddedResource。
    这里只取 text 内容拼接返回。
    """
    result = await session.call_tool(name, args)
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else "(no output)"


# ── Agent loop（与 s01 核心逻辑相同，工具来自 MCP）────────────────────────────

async def agent_loop(session, user_prompt: str) -> None:
    tools = await mcp_list_tools(session)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user_prompt},
    ]

    round_num = 0
    while True:
        round_num += 1
        log.info("── Round %d | history=%d msgs ──", round_num, len(messages))

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            max_tokens=8000,
        )
        elapsed = time.perf_counter() - t0

        message   = response.choices[0].message
        tool_calls = message.tool_calls or []
        stop_reason = response.choices[0].finish_reason
        log.info("LLM <<< (%.2fs) finish=%s tool_calls=%d",
                 elapsed, stop_reason, len(tool_calls))

        # 把 assistant 消息追加回 messages
        msg_dict: dict = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            msg_dict["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        messages.append(msg_dict)

        if not tool_calls:
            # 模型决定停止，输出最终回答
            if message.content:
                print(f"\nAssistant: {message.content}")
            break

        # 执行每个工具调用（通过 MCP session）
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            log.info("TOOL  >>> id=%s name=%s args=%s", tc.id, name, str(args)[:120])

            t0 = time.perf_counter()
            output = await mcp_call_tool(session, name, args)
            elapsed = time.perf_counter() - t0
            log.info("TOOL  <<< id=%s (%.2fs) output_len=%d", tc.id, elapsed, len(output))

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      output,
            })


# ── Main：启动 MCP server 子进程，连接，运行 agent ────────────────────────────

async def main() -> None:
    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters
    except ImportError:
        print("Error: mcp package not installed.\nRun: pip install mcp")
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "List files in the current directory and briefly describe the project."

    # 把内嵌 server 脚本写到临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(MCP_SERVER_SCRIPT)
        server_path = f.name
    os.chmod(server_path, 0o700)   # 仅 owner 可读写执行

    log.info("MCP   server script written to %s", server_path)

    try:
        params = StdioServerParameters(
            command=sys.executable,   # 使用当前 Python 解释器
            args=[server_path],
            env={**os.environ},
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                log.info("MCP   session initialized")
                await agent_loop(session, user_prompt)
    finally:
        os.unlink(server_path)
        log.debug("MCP   server script cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
