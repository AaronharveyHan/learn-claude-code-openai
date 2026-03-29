#!/usr/bin/env python3
"""
s10_team_protocols.py - Team Protocols
Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern. Builds on s09's team messaging.
    Shutdown FSM: pending -> approved | rejected
    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops
    Plan approval FSM: pending -> approved | rejected
    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+
    Trackers: {request_id: {"target|from": name, "status": "pending|..."}}
Key insight: "Same request_id correlation pattern, two domains."
"""
import json
import logging
import os
import subprocess
import threading
import time
import uuid
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
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
SYSTEM = (
    f"You are a team lead at {WORKDIR}. "
    f"Manage teammates with shutdown and plan approval protocols. "
    f"To shut down a teammate: call shutdown_request (sends the request), "
    f"then call check_shutdown_status (polls the result). "
    f"Never call shutdown_response — that tool belongs to teammates only."
)
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval",
    "plan_approval_response",
}
# -- Request trackers: correlate by request_id --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            log.warning("MSG   send: invalid type=%s from=%s to=%s", msg_type, sender, to)
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        log.info("MSG   send: %s -> %s [%s] content=%s", sender, to, msg_type, content[:60])
        return f"Sent {msg_type} to {to}"
    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            log.debug("MSG   read_inbox: %s inbox empty (no file)", name)
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        if messages:
            log.info("MSG   read_inbox: %s drained %d message(s)", name, len(messages))
        else:
            log.debug("MSG   read_inbox: %s inbox empty", name)
        return messages
    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        log.info("MSG   broadcast: %s -> %d teammate(s) content=%s", sender, count, content[:60])
        return f"Broadcast to {count} teammates"
BUS = MessageBus(INBOX_DIR)
# -- TeammateManager with shutdown + plan approval --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}
        self._lock = threading.Lock() 
    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}
    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))
    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None
    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            member = self._find_member(name)
            if member:
                if member["status"] not in ("idle", "shutdown"):
                    log.warning("TEAM  spawn: '%s' is currently %s, cannot spawn", name, member["status"])
                    return f"Error: '{name}' is currently {member['status']}"
                old_status = member["status"]
                member["status"] = "working"
                member["role"] = role
                log.info("TEAM  spawn: reuse '%s' role=%s status %s->working", name, role, old_status)
            else:
                member = {"name": name, "role": role, "status": "working"}
                self.config["members"].append(member)
                log.info("TEAM  spawn: new member '%s' role=%s", name, role)
            self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        log.info("TEAM  spawn: thread started for '%s'", name)
        return f"Spawned '{name}' (role: {role})"
    def _teammate_loop(self, name: str, role: str, prompt: str):
        log.info("[%s] loop started role=%s prompt=%s", name, role, prompt[:60])
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        tools = self._teammate_tools()
        should_exit = False
        first_run = True
        round_num = 0
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            if not first_run and not inbox:
                time.sleep(1)
                continue
            first_run = False
            for msg in inbox:
                log.debug("[%s] inbox msg from=%s type=%s", name,
                          msg.get("from"), msg.get("type"))
                messages.append({"role": "user", "content": json.dumps(msg)})
            if should_exit:
                log.info("[%s] should_exit=True, breaking loop", name)
                break
            round_num += 1
            log.info("[%s] Round %d | history=%d msgs", name, round_num, len(messages))
            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=MODEL, messages=messages,
                    tools=tools, max_tokens=8000,
                )
            except Exception as e:
                log.error("[%s] API error: %s", name, e)
                print(f"[{name}] API error: {e}")
                break
            elapsed = time.perf_counter() - t0
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            stop_reason = response.choices[0].finish_reason
            log.info("[%s] LLM <<< (%.2fs) finish=%s | tool_calls=%d | content_len=%s",
                     name, elapsed, stop_reason, len(tool_calls),
                     len(message.content) if message.content else 0)
            log.debug("[%s] LLM content: %s", name, (message.content or "")[:300])
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
            if not tool_calls:
                log.info("[%s] no tool_calls, continuing to wait for inbox", name)
                continue
            for tool_call in tool_calls:
                assert tool_call.id is not None
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    args = {}
                log.info("[%s] TOOL >>> id=%s name=%s args=%s",
                         name, tool_call.id, tool_call.function.name, args)
                try:
                    output = self._exec(name, tool_call.function.name, args)
                except Exception as e:
                    log.error("[%s] TOOL <<< id=%s EXCEPTION: %s", name, tool_call.id, e)
                    output = f"Error: {e}"
                log.info("[%s] TOOL <<< id=%s output_len=%d", name, tool_call.id, len(output))
                print(f"> {tool_call.function.name}: {output[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": str(output)
                })
                if tool_call.function.name == "shutdown_response" and args.get("approve", False):
                    log.info("[%s] shutdown approved, will exit after this round", name)
                    should_exit = True
        with self._lock:
            member = self._find_member(name)
            if member:
                final_status = "shutdown" if should_exit else "idle"
                member["status"] = final_status
                self._save_config()
        log.info("[%s] loop exited status=%s", name, final_status if member else "unknown")
        # 方案一：主动通知 lead，避免 lead 无限等待
        BUS.send(name, "lead",
                 f"Teammate '{name}' has exited with status={final_status if member else 'unknown'}.",
                 "message")
        log.info("[%s] sent exit notification to lead", name)
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    old = shutdown_requests[req_id]["status"]
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
                    log.info("PROTO shutdown_response: req_id=%s %s->%s by %s",
                             req_id, old, shutdown_requests[req_id]["status"], sender)
                else:
                    log.warning("PROTO shutdown_response: unknown req_id=%s from %s", req_id, sender)
            content = f"Shutdown {'approved' if approve else 'rejected'} by {sender}"
            BUS.send(
                sender, "lead", args.get("reason", content),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            log.info("PROTO plan_approval: req_id=%s from=%s plan=%s", req_id, sender, plan_text[:60])
            BUS.send(
                sender, "lead", plan_text, "plan_approval",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        return f"Unknown tool: {tool_name}"
    def _teammate_tools(self) -> list:
        # these base tools are unchanged from s02
        return [
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
                    "path": {"type": "string"}
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
            "name": "send_message",
            "description": "Send message to a teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {
                        "type": "string",
                        "enum": list(VALID_MSG_TYPES),
                        "description": "Type of message: e.g. task, info, alert"
                    }
                },
                "required": ["to", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "Read and drain your inbox.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_response",
            "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "reason": {"type": "string"}
                },
                "required": ["request_id", "approve"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Submit a plan for lead approval. Provide plan text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"}
                },
                "required": ["plan"]
            }
        }
    },
]
    def is_alive(self, name: str) -> bool:
        """方案三：检查 teammate 线程是否仍在运行。"""
        thread = self.threads.get(name)
        return thread is not None and thread.is_alive()
    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            # 方案三：同时显示 config 状态和线程存活情况
            alive = self.is_alive(m["name"])
            thread_info = "thread=alive" if alive else "thread=dead"
            lines.append(f"  {m['name']} ({m['role']}): {m['status']} [{thread_info}]")
        return "\n".join(lines)
    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]
TEAM = TeammateManager(TEAM_DIR)
# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        log.warning("BLOCKED dangerous command: %s", command)
        return "Error: Dangerous command blocked"
    log.debug("BASH  >>> %s", command)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
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
def _run_read(path: str, limit: int = None) -> str:
    log.debug("READ  >>> path=%s limit=%s", path, limit)
    t0 = time.perf_counter()
    try:
        lines = _safe_path(path).read_text().splitlines()
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
def _run_write(path: str, content: str) -> str:
    log.debug("WRITE >>> path=%s bytes=%d", path, len(content))
    t0 = time.perf_counter()
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        result = f"Wrote {len(content)} bytes"
        elapsed = time.perf_counter() - t0
        log.debug("WRITE <<< (%.2fs) %s", elapsed, result)
        return result
    except Exception as e:
        log.error("WRITE <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
def _run_edit(path: str, old_text: str, new_text: str) -> str:
    log.debug("EDIT  >>> path=%s old_len=%d new_len=%d", path, len(old_text), len(new_text))
    t0 = time.perf_counter()
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            log.warning("EDIT  <<< Text not found in %s", path)
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        result = f"Edited {path}"
        elapsed = time.perf_counter() - t0
        log.debug("EDIT  <<< (%.2fs) %s", elapsed, result)
        return result
    except Exception as e:
        log.error("EDIT  <<< EXCEPTION path=%s: %s", path, e)
        return f"Error: {e}"
# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    log.info("PROTO shutdown_request: req_id=%s -> '%s' status=pending", req_id, teammate)
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        log.warning("PROTO plan_review: unknown request_id=%s", request_id)
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    log.info("PROTO plan_review: req_id=%s from=%s status=%s feedback=%s",
             request_id, req["from"], req["status"], feedback[:60])
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"
def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        result = shutdown_requests.get(request_id, {"error": "not found"})
    log.debug("PROTO check_shutdown_status: req_id=%s result=%s", request_id, result)
    return json.dumps(result)
# -- Lead tool dispatch (12 tools) --
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "check_shutdown_status": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}
# these base tools are unchanged from s02
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
            "name": "spawn_teammate",
            "description": "Spawn a persistent teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"}
                },
                "required": ["name", "role", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List all teammates.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {
                        "type": "string",
                        "enum": list(VALID_MSG_TYPES)
                    }
                },
                "required": ["to", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "Read and drain the lead's inbox.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Send a message to all teammates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "teammate": {"type": "string"}
                },
                "required": ["teammate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_shutdown_status",
            "description":
                "Poll the status of a shutdown request by request_id. "
                "Returns 'pending', 'approved', or 'rejected'. "
                "Use this after sending shutdown_request to see if the teammate responded.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string", "description": "The request_id returned by shutdown_request."}
                },
                "required": ["request_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "feedback": {"type": "string"}
                },
                "required": ["request_id", "approve"]
            }
        }
    },
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
        snippet = (msg.get("content") or "")[:80]
        return f"system     content={snippet!r}"
    return repr(msg)[:120]

def _log_msg_append(messages: list, new_msg: dict) -> None:
    """每次 messages.append() 后调用，记录新增消息及当前总长度。"""
    log.debug("MSGS  +1 [total=%d]  %s", len(messages), _msg_summary(new_msg))

# ── Agent loop (lead) ─────────────────────────────────────────────────────────
INBOX_EMPTY_WARN = 10   # 方案二：连续空 inbox 超过此轮数注入超时提醒

def agent_loop(messages: list):
    round_num = 0
    empty_inbox_rounds = 0
    while True:
        round_num += 1
        log.info("── Round %d | history=%d msgs ──", round_num, len(messages))
        inbox = BUS.read_inbox("lead")
        if inbox:
            empty_inbox_rounds = 0
            log.info("LEAD  inbox: %d message(s) received", len(inbox))
            inbox_msg = {
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            }
            ack_msg = {
                "role": "assistant",
                "content": "Noted inbox messages.",
            }
            messages.append(inbox_msg)
            _log_msg_append(messages, inbox_msg)
            messages.append(ack_msg)
            _log_msg_append(messages, ack_msg)
        else:
            empty_inbox_rounds += 1
            # 方案二：连续空 inbox 达到阈值，注入超时提醒让 LLM 自主决策
            if empty_inbox_rounds >= INBOX_EMPTY_WARN:
                log.warning("LEAD  inbox empty for %d consecutive rounds, injecting timeout hint",
                            empty_inbox_rounds)
                timeout_msg = {
                    "role": "user",
                    "content": (
                        f"<system>Inbox has been empty for {empty_inbox_rounds} consecutive rounds. "
                        f"A teammate may have exited without responding. "
                        f"Consider calling list_teammates to check thread status, "
                        f"or stop waiting and proceed.</system>"
                    ),
                }
                messages.append(timeout_msg)
                _log_msg_append(messages, timeout_msg)
                empty_inbox_rounds = 0
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

if __name__ == "__main__":
    log.info("Agent started. Model=%s  LogFile=%s", MODEL, LOG_FILE)
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        log.info("USER  >>> %s", query)
        user_msg = {"role": "user", "content": query}
        history.append(user_msg)
        _log_msg_append(history, user_msg)
        response_content = agent_loop(history)
        print(response_content)
    log.info("Agent exited.")