#!/usr/bin/env python3
"""
s02_tool_use.py - Tools
The agent loop from s01 didn't change. We just added tools to the array
and a dispatch map to route calls.
    +----------+      +-------+      +------------------+
    |   User   | ---> |  LLM  | ---> | Tool Dispatch    |
    |  prompt  |      |       |      | {                |
    +----------+      +---+---+      |   bash: run_bash |
                          ^          |   read: run_read |
                          |          |   write: run_wr  |
                          +----------+   edit: run_edit |
                          tool_result| }                |
                                     +------------------+
Key insight: "The loop didn't change at all. I just added tools."
"""
import os
import json
import time
import logging
import tempfile
import subprocess
from openai import OpenAI 
from pathlib import Path
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
 
    # 终端 handler（彩色，DEBUG+）
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColorFormatter(fmt, datefmt))
 
    # 文件 handler（纯文本，DEBUG+）
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
                api_key=os.getenv("llm_api_key"),          # None 时自动读 OPENAI_API_KEY 或 DASHSCOPE_API_KEY
                base_url=os.getenv("llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
MODEL = os.getenv("llm_model","qwen3.5-plus")
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."
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
        elapsed = time.perf_counter() - t0
        log.debug("BASH  <<< (%.2fs, rc=%d) %s", elapsed, r.returncode, out[:300])
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log.error("BASH  <<< TIMEOUT (120s): %s", command)
        return "Error: Timeout (120s)"
    except Exception as e:
        log.error("BASH  <<< EXCEPTION: %s", e)
        return f"Error: {e}"
    
def run_read(path: str, limit: int = None) -> str:
    log.debug("READ  >>> path=%s limit=%s", path, limit)
    t0 = time.perf_counter()
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        total_lines = len(lines)
        if limit and limit < total_lines:
            lines = lines[:limit] + [f"... ({total_lines - limit} more lines)"]
        result = "\n".join(lines)[:50000]
        elapsed = time.perf_counter() - t0
        log.debug("READ  <<< (%.2fs) total_lines=%d returned_chars=%d", elapsed, total_lines, len(result))
        return result
    except Exception as e:
        log.error("READ  <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
def run_write(path: str, content: str) -> str:
    log.debug("WRITE >>> path=%s bytes=%d", path, len(content))
    t0 = time.perf_counter()
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=fp.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, fp)
        except:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        result = f"Wrote {len(content)} bytes to {path}"
        elapsed = time.perf_counter() - t0
        log.debug("WRITE <<< (%.2fs) %s", elapsed, result)
        return result
    except Exception as e:
        log.error("WRITE <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
def run_edit(path: str, old_text: str, new_text: str) -> str:
    log.debug("EDIT  >>> path=%s old_len=%d new_len=%d", path, len(old_text), len(new_text))
    t0 = time.perf_counter()
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            log.warning("EDIT  <<< Text not found in %s", path)
            return f"Error: Text not found in {path}"
        new_content = content.replace(old_text, new_text, 1)
        fd, tmp = tempfile.mkstemp(dir=fp.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(tmp, fp)
        except:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        result = f"Edited {path}"
        elapsed = time.perf_counter() - t0
        log.debug("EDIT  <<< (%.2fs) %s", elapsed, result)
        return result
    except Exception as e:
        log.error("EDIT  <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    }
]
# ── Helpers ───────────────────────────────────────────────────────────────────
def _msg_summary(msg: dict) -> str:
    """返回单条 message 的一行摘要，用于 messages 变化日志。"""
    role = msg.get("role", "?")
    if role == "assistant":
        tcs = msg.get("tool_calls", [])
        if tcs:
            names = [tc["function"]["name"] for tc in tcs]
            return f"assistant  tool_calls={names}"
        content = (msg.get("content") or "")[:80]
        return f"assistant  content={content!r}"
    if role == "tool":
        tid = msg.get("tool_call_id", "")
        snippet = str(msg.get("content", ""))[:80]
        return f"tool       id={tid}  content={snippet!r}"
    if role == "user":
        snippet = str(msg.get("content", ""))[:80]
        return f"user       content={snippet!r}"
    if role == "system":
        return "system     (prompt)"
    return repr(msg)[:120]

def _log_msg_append(messages: list, new_msg: dict) -> None:
    """每次 messages.append() 后调用，记录新增消息及当前总长度。"""
    log.debug("MSGS  +1 [total=%d]  %s", len(messages), _msg_summary(new_msg))

# ── Agent loop ────────────────────────────────────────────────────────────────

def agent_loop(messages: list):
    round_num = 0
    while True:
        round_num += 1
        log.info("── Round %d | history=%d msgs ──", round_num, len(messages))

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        elapsed = time.perf_counter() - t0
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        stop_reason = response.choices[0].finish_reason
        log.info(
            "LLM   <<< (%.2fs) finish=%s | tool_calls=%d | content_len=%s",
            elapsed, stop_reason, len(tool_calls),
            len(message.content) if message.content else 0,
        )
        log.debug("LLM content: %s", (message.content or "")[:500])
        # Append assistant turn
        assistant_msg = {
            "role": "assistant",
            "content": message.content or ""  # 或者条件过滤掉 content 键
        }
        if tool_calls:   # ✅ 只有非空才加
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)
        _log_msg_append(messages, assistant_msg)
        # If the model didn't call a tool, we're done
        if not tool_calls:
            log.info("── Agent finished after %d round(s) ──", round_num)
            return message.content
        # Execute each tool call, collect results
        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            log.info("TOOL  >>> id=%s  cmd=%s", tool_call.id, args)
            output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            log.info("TOOL  <<< id=%s  output_len=%d", tool_call.id, len(output))
            print(f"> {tool_call.function.name}: {output[:200]}")
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(output)
            }
            messages.append(tool_msg)
            _log_msg_append(messages, tool_msg)
if __name__ == "__main__":
    log.info("Agent started. Model=%s  LogFile=%s", MODEL, LOG_FILE)
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        log.info("USER  >>> %s", query)
        user_msg = {"role": "user", "content": query}
        history.append(user_msg)
        _log_msg_append(history, user_msg)
        response_content = agent_loop(history)
        print(response_content)
    log.info("Agent exited.")