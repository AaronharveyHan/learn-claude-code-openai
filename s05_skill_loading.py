#!/usr/bin/env python3
"""
s05_skill_loading.py - Skills
Two-layer skill injection that avoids bloating the system prompt:
    Layer 1 (cheap): skill names in system prompt (~100 tokens/skill)
    Layer 2 (on demand): full skill body in tool_result
    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md
    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: metadata only
    |   - code-review: Review code...      |
    +--------------------------------------+
    When model calls load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: full body
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                             |
    +--------------------------------------+
Key insight: "Don't put everything in the system prompt. Load on demand."
"""
import os
import json
import re
import time
import logging
import tempfile
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
SKILLS_DIR = WORKDIR / "skills"
# -- SkillLoader: scan skills/<name>/SKILL.md with YAML frontmatter --
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()
    def _load_all(self):
        if not self.skills_dir.exists():
            log.warning("SKILL _load_all: skills_dir not found: %s", self.skills_dir)
            return
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            if not f.resolve().is_relative_to(self.skills_dir.resolve()):
                log.warning("SKILL _load_all: skipping symlink outside skills_dir: %s", f)
                continue
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}
            log.debug("SKILL loaded: name=%s path=%s body_len=%d", name, f, len(body))
        log.info("SKILL _load_all: %d skill(s) loaded", len(self.skills))
    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()
    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)
    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        log.debug("SKILL >>> load name=%s", name)
        skill = self.skills.get(name)
        if not skill:
            log.warning("SKILL <<< unknown name=%s available=%s", name, list(self.skills.keys()))
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        result = f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
        log.debug("SKILL <<< loaded name=%s body_len=%d", name, len(skill["body"]))
        return result
SKILL_LOADER = SkillLoader(SKILLS_DIR)
# Layer 1: skill metadata injected into system prompt
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.
Skills available:
{SKILL_LOADER.get_descriptions()}"""
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
def run_read(path: str, limit_lines: int = None) -> str:
    """
    Args:
        limit_lines: 最多读取的行数，None 表示读全部，0 表示不读内容
    """
    log.debug("READ  >>> path=%s limit_lines=%s", path, limit_lines)
    t0 = time.perf_counter()
    try:
        lines = safe_path(path).read_text().splitlines()
        total_lines = len(lines)
        if limit_lines is not None:          # 明确区分 0 和 None
            if limit_lines == 0:
                log.debug("READ  <<< path=%s total_lines=%d (no content read)", path, total_lines)
                return f"(file has {total_lines} lines, no content read)"
            if limit_lines < total_lines:
                lines = lines[:limit_lines] + [f"... ({total_lines - limit_lines} more)"]
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
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit_lines")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
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
                    "limit_lines": {"type": "integer"}
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
            "name": "load_skill",
            "description": "Load specialized knowledge by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to load"
                    }
                },
                "required": ["name"]
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
        if not tool_calls:
            log.info("── Agent finished after %d round(s) ──", round_num)
            return message.content

        for tool_call in tool_calls:
            assert tool_call.id is not None
            args = json.loads(tool_call.function.arguments)
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

if __name__ == "__main__":
    log.info("Agent started. Model=%s  LogFile=%s", MODEL, LOG_FILE)
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
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