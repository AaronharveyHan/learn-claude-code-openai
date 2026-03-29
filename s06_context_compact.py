#!/usr/bin/env python3
"""
s06_context_compact.py - Compact
Three-layer compression pipeline so the agent can work forever:
    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.
Key insight: "The agent can forget strategically and keep working forever."
"""
import os
import json
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
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks.When calling compact, always provide a focus describing what must be preserved."
MODEL_CONTEXT_WINDOW = 131072   # qwen3.5-plus 128K
MAX_TOKENS = 8000               # 单次生成上限
ESTIMATE_MARGIN = 0.8           # chars//4 粗估，留 20% 误差缓冲
THRESHOLD = int((MODEL_CONTEXT_WINDOW - MAX_TOKENS) * ESTIMATE_MARGIN)
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4
# -- Layer 1: micro_compact - replace old tool results with placeholders --
def micro_compact(messages: list) -> list:
    # Collect (msg_index, part_index, tool_result_dict) for all tool_result entries
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] == "tool":
            tool_results.append((msg_idx, msg))

    if len(tool_results) <= KEEP_RECENT:
        log.debug("MICRO tool_results=%d <= KEEP_RECENT=%d, skip", len(tool_results), KEEP_RECENT)
        return messages
    to_clear = tool_results[:-KEEP_RECENT]
    cleared = 0
    for _, result in to_clear:
        if isinstance(result.get("content"), str) and len(result["content"]) > 100:
            tool_name = result.get("name", "unknown")
            result["content"] = f"[Previous: used {tool_name}]"
            cleared += 1
    log.debug("MICRO compacted %d/%d old tool_results", cleared, len(to_clear))
    return messages
# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list, focus: str = None) -> list:
    tokens_before = estimate_tokens(messages)
    log.info("AUTO  compact triggered | msgs=%d est_tokens=%d focus=%s",
             len(messages), tokens_before, focus)
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    log.info("AUTO  transcript saved: %s", transcript_path)
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]

    base_prompt = (
        "Summarize this conversation for continuity. Include:\n"
        "1) What was accomplished\n"
        "2) Current state\n"
        "3) Key decisions made\n"
        "4) Include key tool calls and their outputs.\n"
        "5) Available tools and how they were used.\n"
        "6) What tools should be used next.\n"
    )

    if focus:
        base_prompt += f"\nFocus especially on: {focus}\n"

    prompt = base_prompt + "\n\n" + conversation_text

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    elapsed = time.perf_counter() - t0
    summary = response.choices[0].message.content or ""
    log.info("AUTO  summarized (%.2fs) summary_len=%d", elapsed, len(summary))
    # Replace all messages with compressed summary
    new_messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]
    tokens_after = estimate_tokens(new_messages)
    log.info("AUTO  done | msgs %d->%d est_tokens %d->%d",
             len(messages), len(new_messages), tokens_before, tokens_after)
    return new_messages
# -- Tool implementations --
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
        lines = safe_path(path).read_text().splitlines()
        total_lines = len(lines)
        if limit and limit < total_lines:
            lines = lines[:limit] + [f"... ({total_lines - limit} more)"]
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
        fp.write_text(content)
        result = f"Wrote {len(content)} bytes"
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
        fp.write_text(content.replace(old_text, new_text, 1))
        result = f"Edited {path}"
        elapsed = time.perf_counter() - t0
        log.debug("EDIT  <<< (%.2fs) %s", elapsed, result)
        return result
    except Exception as e:
        log.error("EDIT  <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact":    lambda **kw: "Manual compression requested.",
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
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary"
                    }
                }
                # 👇 注意：这里没有 required（和你原来一致）
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
        est_tokens = estimate_tokens(messages)
        log.info("── Round %d | history=%d msgs | est_tokens=%d ──",
                 round_num, len(messages), est_tokens)
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if est_tokens > THRESHOLD:
            log.warning("TOKENS %d > THRESHOLD %d, triggering auto_compact", est_tokens, THRESHOLD)
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
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
            "content": message.content or ""
        }
        if tool_calls:
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
        focus = None
        non_compact_calls = [tc for tc in tool_calls if tc.function.name != "compact"]
        compact_calls     = [tc for tc in tool_calls if tc.function.name == "compact"]
        log.debug("TOOL  dispatch: %d normal + %d compact call(s)",
                  len(non_compact_calls), len(compact_calls))
        for tool_call in non_compact_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            log.info("TOOL  >>> id=%s  name=%s  args=%s", tool_call.id, tool_call.function.name, args)
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            except Exception as e:
                log.error("TOOL  <<< id=%s EXCEPTION: %s", tool_call.id, e)
                output = f"Error: {e}"
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
        for tool_call in compact_calls:
            # 最后处理 compact
            focus = json.loads(tool_call.function.arguments or "{}").get("focus")
            log.info("TOOL  >>> id=%s  name=compact  focus=%s", tool_call.id, focus)
            ack_msg = {"role": "tool", "tool_call_id": tool_call.id,
                       "name": "compact", "content": "OK"}
            messages.append(ack_msg)
            _log_msg_append(messages, ack_msg)
        # Layer 3: compact always runs last, after all other tools
        if compact_calls:
            focus_parts = [
                json.loads(tc.function.arguments or "{}").get("focus", "")
                for tc in compact_calls
            ]
            focus = "; ".join(f for f in focus_parts if f) or None
            log.info("COMPACT manual triggered | focus=%s", focus)
            print("[manual compact]")
            messages[:] = auto_compact(messages, focus)
            # auto_compact 替换了整个 messages，直接进入下一轮循环

if __name__ == "__main__":
    log.info("Agent started. Model=%s  LogFile=%s", MODEL, LOG_FILE)
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
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
