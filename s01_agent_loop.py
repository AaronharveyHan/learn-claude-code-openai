#!/usr/bin/env python3
"""
s01_agent_loop.py - The Agent Loop
The entire secret of an AI coding agent in one pattern:
    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results
    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)
This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""
import os
import json
import logging
import subprocess
import time
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

client = OpenAI(
                api_key=os.getenv("llm_api_key"),          # None 时自动读 OPENAI_API_KEY 或 DASHSCOPE_API_KEY
                base_url=os.getenv("llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
MODEL = os.getenv("llm_model","qwen3.5-plus")
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
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
    }
]
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        log.warning("BLOCKED dangerous command: %s", command)
        return "Error: Dangerous command blocked"
    log.debug("BASH  >>> %s", command)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
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

# -- The core pattern: a while loop that calls tools until the model stops --
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
            cmd = args["command"]
            log.info("TOOL  >>> id=%s  cmd=%s", tool_call.id, cmd)
            print(f"\033[33m$ {cmd}\033[0m")
            output = run_bash(cmd)
            log.info("TOOL  <<< id=%s  output_len=%d", tool_call.id, len(output))
            print(output[:200])
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
            query = input("\033[36ms01 >> \033[0m")
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