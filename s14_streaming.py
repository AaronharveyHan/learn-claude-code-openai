#!/usr/bin/env python3
"""
s14_streaming.py - Streaming Output

All previous scripts wait for the complete response before doing anything.
Streaming (stream=True) lets the model send tokens as they are generated,
so the user sees output immediately instead of waiting several seconds.

The agent loop is identical to s02. Only two things change:

    1. API call:  stream=True  → returns an iterator of chunks
    2. Response:  accumulate chunks → reconstruct the full message

    Non-streaming (s01-s13):
        response = client.chat.completions.create(...)   # blocks until done
        message  = response.choices[0].message

    Streaming (s14):
        stream   = client.chat.completions.create(..., stream=True)
        for chunk in stream:                             # yields token-by-token
            ...accumulate...
        message  = reconstruct(accumulated_chunks)

The hard part is tool_calls: each chunk carries a *delta* that must be
assembled into the full call. Content is simple concatenation; tool call
arguments also arrive as fragments and need the same treatment.

    Chunk anatomy:
        chunk.choices[0].delta.content          → text fragment (or None)
        chunk.choices[0].delta.tool_calls       → list of ToolCallDelta
            .index                              → which parallel tool call
            .id                                 → call ID (first chunk only)
            .function.name                      → tool name (first chunk only)
            .function.arguments                 → argument JSON fragment
        chunk.choices[0].finish_reason          → None | "stop" | "tool_calls"

Run:
    python s14_streaming.py
    python s14_streaming.py "list files and show the first 10 lines of s01"
"""
import os
import json
import sys
import tempfile
import time
import logging
import subprocess
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Logger setup ──────────────────────────────────────────────────────────────
LOG_FILE = "agent_debug.log"

class ColorFormatter(logging.Formatter):
    """终端彩色输出，文件输出保持纯文本。"""
    COLORS = {
        logging.DEBUG:    "\033[90m",
        logging.INFO:     "\033[36m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[35m",
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
MODEL  = os.getenv("llm_model", "qwen-plus")
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."

# ── Tools（与 s02 完全相同）──────────────────────────────────────────────────
def _atomic_write(path: Path, content: str) -> None:
    """先写临时文件再 rename，防止崩溃时文件损坏。"""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except:
        try: os.unlink(tmp)
        except OSError: pass
        raise

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        log.warning("BLOCKED dangerous command: %s", command)
        return "Error: Dangerous command blocked"
    log.debug("BASH  >>> %s", command)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        log.debug("BASH  <<< (%.2fs, rc=%d) %s", time.perf_counter() - t0, r.returncode, out[:200])
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except Exception as e:
        return f"Error: {e}"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        total = len(lines)
        if limit and limit < total:
            lines = lines[:limit] + [f"... ({total - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(fp, content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        _atomic_write(fp, content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read file contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace old_text with new_text in a file.",
        "parameters": {"type": "object",
                       "properties": {"path":     {"type": "string"},
                                      "old_text": {"type": "string"},
                                      "new_text": {"type": "string"}},
                       "required": ["path", "old_text", "new_text"]}}},
]

# ── 核心新增：流式调用 + chunk 累积 ──────────────────────────────────────────

def stream_llm(messages: list) -> tuple[str, list[dict]]:
    """
    发起流式请求，实时打印 content，累积 tool_calls。

    返回值：
        content    (str)        — 模型生成的完整文本
        tool_calls (list[dict]) — 完整的工具调用列表（与非流式格式相同）

    tool_call delta 累积规则：
        - 同一工具调用的所有 chunk 共享同一个 index
        - id / name 只在第一个 chunk 出现，后续为 None
        - arguments 是 JSON 字符串的逐片追加
    """
    t0 = time.perf_counter()
    log.info("LLM  >>> stream=True history=%d msgs", len(messages))

    # acc[index] = {"id": ..., "name": ..., "arguments": ...}
    acc: dict[int, dict] = {}
    content_buf = ""
    finish_reason = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=8000,
        stream=True,          # ← 唯一与 s02 不同的参数
    )

    print("\n\033[32mAssistant:\033[0m ", end="", flush=True)

    for chunk in stream:
        choice = chunk.choices[0]
        delta  = choice.delta
        finish_reason = choice.finish_reason or finish_reason

        # ── 文本 fragment：实时打印 ──────────────────────────────
        if delta.content:
            print(delta.content, end="", flush=True)
            content_buf += delta.content

        # ── tool_call fragment：按 index 累积 ───────────────────
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in acc:
                    acc[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    acc[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    acc[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    acc[idx]["arguments"] += tc.function.arguments

    elapsed = time.perf_counter() - t0
    if content_buf:
        print()   # 文本结束后换行

    # 将累积结果转换为标准 tool_calls 格式
    tool_calls = [
        {
            "id":       acc[i]["id"],
            "type":     "function",
            "function": {"name": acc[i]["name"], "arguments": acc[i]["arguments"]},
        }
        for i in sorted(acc)
    ]

    log.info("LLM  <<< (%.2fs) finish=%s content_len=%d tool_calls=%d",
             elapsed, finish_reason, len(content_buf), len(tool_calls))
    for tc in tool_calls:
        log.debug("LLM  tool_call name=%s args=%s",
                  tc["function"]["name"], tc["function"]["arguments"][:120])

    return content_buf, tool_calls


# ── Agent loop（与 s02 相同，只是把 API 调用换成 stream_llm）────────────────

def agent_loop(messages: list) -> str:
    round_num = 0
    while True:
        round_num += 1
        log.info("── Round %d | history=%d msgs ──", round_num, len(messages))

        # 流式调用：实时打印内容，返回累积后的完整结果
        content, tool_calls = stream_llm(messages)

        # 把 assistant 消息追加回 messages（格式与非流式完全相同）
        assistant_msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # 模型不再调用工具，循环结束
            return content

        # 执行每个工具调用
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args   = json.loads(tc["function"]["arguments"] or "{}")
                output = TOOL_HANDLERS[name](**args)
            except Exception as e:
                log.error("TOOL  <<< EXCEPTION name=%s: %s", name, e)
                output = f"Error: {e}"

            log.info("TOOL  name=%s output_len=%d", name, len(output))
            messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      output,
            })


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Agent started (streaming). Model=%s  LogFile=%s", MODEL, LOG_FILE)
    history = [{"role": "system", "content": SYSTEM}]

    # 支持命令行直接传入任务，或进入交互模式
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        log.info("USER  >>> %s", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
    else:
        while True:
            try:
                query = input("\n\033[36ms14 >> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            log.info("USER  >>> %s", query)
            history.append({"role": "user", "content": query})
            agent_loop(history)

    log.info("Agent exited.")
