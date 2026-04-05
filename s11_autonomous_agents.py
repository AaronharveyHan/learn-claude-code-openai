#!/usr/bin/env python3
"""
s11_autonomous_agents.py - Autonomous Agents
Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.
    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown
    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"
Key insight: "The agent finds work itself."
"""
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

_shutdown = threading.Event()   # C1: 优雅退出信号，主程序退出时 set()
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

WORKDIR = Path.cwd()
client = OpenAI(
                api_key=os.getenv("llm_api_key"),          # None 时自动读 OPENAI_API_KEY 或 DASHSCOPE_API_KEY
                base_url=os.getenv("llm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )
MODEL = os.getenv("llm_model","qwen3.5-plus")
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

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
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}
# -- Request trackers --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()
# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()   # 保护 _locks 字典本身
    def _get_lock(self, name: str) -> threading.Lock:
        # 为每个收件人懒初始化一把锁
        with self._meta_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]
    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            log.warning("BUS  send invalid type=%r from=%s to=%s", msg_type, sender, to)
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
        # 加锁后再写，与 read_inbox 互斥
        with self._get_lock(to):
            with open(inbox_path, "a") as f:
                f.write(json.dumps(msg) + "\n")
        log.info("BUS  send type=%s from=%s to=%s content=%r", msg_type, sender, to, content[:80])
        return f"Sent {msg_type} to {to}"
    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        # 读+清空必须在同一把锁内原子完成
        with self._get_lock(name):
            if not inbox_path.exists():
                return []
            raw = inbox_path.read_text()
            inbox_path.write_text("")          # 清空紧跟读，中间不会被 send 插入
        messages = []
        for line in raw.strip().splitlines():
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                       # 顺手处理损坏行，不让解析异常丢掉整批消息
        if messages:
            log.debug("BUS  read_inbox name=%s count=%d", name, len(messages))
        return messages
    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        log.info("BUS  broadcast from=%s count=%d content=%r", sender, count, content[:80])
        return f"Broadcast to {count} teammates"
BUS = MessageBus(INBOX_DIR)
# ── Message logging helpers ────────────────────────────────────────────────────
def _msg_summary(msg: dict) -> str:
    role = msg.get("role", "?")
    if role == "system":
        content = msg.get("content", "")
        return f"system: {content[:60]!r}"
    if role == "tool":
        return f"tool[{msg.get('name','?')}]: {str(msg.get('content',''))[:60]!r}"
    if role == "assistant":
        tcs = msg.get("tool_calls", [])
        if tcs:
            names = [tc["function"]["name"] for tc in tcs]
            return f"assistant: tool_calls={names}"
        return f"assistant: {str(msg.get('content',''))[:60]!r}"
    return f"{role}: {str(msg.get('content',''))[:60]!r}"

def _log_msg_append(messages: list, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    log.debug("%sMSG  +[%d] %s", tag, len(messages), _msg_summary(messages[-1]))

# -- Task board scanning --
def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    log.debug("TASK scan_unclaimed count=%d", len(unclaimed))
    return unclaimed
def claim_task(task_id: int | str, owner: str) -> str:
    # 统一转为 int，兼容 JSON 字符串 id（如 "3"）和整数 id（如 3）
    try:
        task_id = int(task_id)
    except (TypeError, ValueError):
        log.warning("TASK claim invalid task_id=%r owner=%s", task_id, owner)
        return f"Error: Invalid task_id '{task_id}', must be an integer"

    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            log.warning("TASK claim task_id=%d not found owner=%s", task_id, owner)
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        # 双重校验：文件名和 JSON 内 id 要一致
        if int(task.get("id", -1)) != task_id:
            log.warning("TASK claim id mismatch task_%d.json", task_id)
            return f"Error: Task file id mismatch for task_{task_id}.json"
        if task.get("status") != "pending" or task.get("owner"):
            log.warning("TASK claim already claimed task_id=%d by=%s", task_id, task.get("owner"))
            return f"Error: Task {task_id} already claimed by {task.get('owner')}"
        task["owner"] = owner
        task["status"] = "in_progress"
        _atomic_write(path, json.dumps(task, indent=2))
    log.info("TASK claim task_id=%d owner=%s subject=%r", task_id, owner, task.get("subject","")[:60])
    return f"Claimed task #{task_id} for {owner}"
# -- Identity re-injection after compression --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }
# -- Autonomous TeammateManager --
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
        _atomic_write(self.config_path, json.dumps(self.config, indent=2))
    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None
    def _set_status(self, name: str, status: str):
        with self._lock:
            member = self._find_member(name)
            if member:
                member["status"] = status
                self._save_config()
    def spawn(self, name: str, role: str, prompt: str) -> str:
        with self._lock:
            member = self._find_member(name)
            if member:
                if member["status"] not in ("idle", "shutdown"):
                    log.warning("TEAM spawn '%s' already %s", name, member["status"])
                    return f"Error: '{name}' is currently {member['status']}"
                log.info("TEAM spawn reuse name=%s role=%s status=%s->working", name, role, member["status"])
                member["status"] = "working"
                member["role"] = role
            else:
                log.info("TEAM spawn new name=%s role=%s", name, role)
                member = {"name": name, "role": role, "status": "working"}
                self.config["members"].append(member)
            self._save_config()
        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=False,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"
    def shutdown(self, timeout: float = 5.0) -> None:
        """Signal all threads to stop and wait for them to finish."""
        _shutdown.set()
        for name, t in list(self.threads.items()):
            t.join(timeout=timeout)
            if t.is_alive():
                log.warning("TEAM shutdown: '%s' still alive after %.1fs", name, timeout)
    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "system", "content": sys_prompt},{"role": "user", "content": prompt}]
        _log_msg_append(messages, name)
        _log_msg_append(messages[1:], name)
        tools = self._teammate_tools()
        log.info("[%s] loop started role=%s team=%s", name, role, team_name)
        work_cycle = 0
        while not _shutdown.is_set():
            work_cycle += 1
            log.debug("[%s] WORK cycle=%d msgs=%d", name, work_cycle, len(messages))
            # -- WORK PHASE: standard agent loop --
            for round_num in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        log.info("[%s] shutdown_request received, exiting", name)
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                    _log_msg_append(messages, name)
                log.debug("[%s] Round %d msgs=%d", name, round_num + 1, len(messages))
                t0 = time.perf_counter()
                try:
                    response = client.chat.completions.create(
                        model=MODEL, messages=messages,
                        tools=tools, max_tokens=8000,
                    )
                except Exception as e:
                    log.error("[%s] LLM exception: %s", name, e)
                    self._set_status(name, "idle")
                    return
                elapsed = time.perf_counter() - t0
                message = response.choices[0].message
                tool_calls = message.tool_calls or []
                log.debug("[%s] LLM <<< elapsed=%.2fs stop=%s tool_calls=%d",
                          name, elapsed, response.choices[0].finish_reason, len(tool_calls))
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
                _log_msg_append(messages, name)
                if not tool_calls:
                    log.info("[%s] WORK done (no tool_calls) round=%d", name, round_num + 1)
                    return message.content
                idle_requested = False
                for tool_call in tool_calls:
                    assert tool_call.id is not None
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except Exception:
                        args = {}
                    log.debug("[%s] TOOL >>> %s args=%s", name, tool_call.function.name, str(args)[:120])
                    t1 = time.perf_counter()
                    if tool_call.function.name == "idle":
                            idle_requested = True
                            output = "Entering idle phase. Will poll for new tasks."
                    else:
                        try:
                            output = self._exec(name, tool_call.function.name, args)
                        except Exception as e:
                            log.error("[%s] TOOL <<< %s EXCEPTION: %s", name, tool_call.function.name, e)
                            output = f"Error: {e}"
                    log.debug("[%s] TOOL <<< %s elapsed=%.2fs out=%r",
                              name, tool_call.function.name, time.perf_counter() - t1, str(output)[:80])
                    print(f"> {tool_call.function.name}: {output[:200]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(output)
                    })
                    _log_msg_append(messages, name)
                if idle_requested:
                    log.info("[%s] idle requested, entering IDLE phase", name)
                    break

            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            self._set_status(name, "idle")
            log.info("[%s] IDLE phase started polls=%d interval=%ds",
                     name, IDLE_TIMEOUT // max(POLL_INTERVAL, 1), POLL_INTERVAL)
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for poll_i in range(polls):
                _shutdown.wait(timeout=POLL_INTERVAL)   # 可被 shutdown 信号立即唤醒
                inbox = BUS.read_inbox(name)
                if inbox:
                    log.info("[%s] IDLE inbox msg count=%d at poll=%d", name, len(inbox), poll_i + 1)
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            log.info("[%s] shutdown_request in IDLE, exiting", name)
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                        _log_msg_append(messages, name)
                    resume = True
                    break
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    task_id = task.get("id")
                    if task_id is None:
                        # task 文件没有 id 字段，跳过这个损坏的任务
                        log.warning("[%s] IDLE unclaimed task missing id field, skipping", name)
                        pass
                    else:
                        _ = claim_task(task["id"], name)
                        log.info("[%s] IDLE auto-claimed task_id=%s subject=%r at poll=%d",
                                 name, task["id"], task.get("subject","")[:60], poll_i + 1)
                        task_prompt = (
                            f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                            f"{task.get('description', '')}</auto-claimed>"
                        )
                        if len(messages) <= 3:
                            messages.insert(1, make_identity_block(name, role, team_name))
                            messages.insert(2, {"role": "assistant", "content": f"I am {name}. Continuing."})
                            log.debug("[%s] IDLE identity re-injected (short context)", name)
                        messages.append({"role": "user", "content": task_prompt})
                        _log_msg_append(messages, name)
                        messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                        _log_msg_append(messages, name)
                        resume = True
                        break
                else:
                    log.debug("[%s] IDLE poll=%d no inbox no unclaimed tasks", name, poll_i + 1)
            if not resume:
                log.info("[%s] IDLE timeout after %ds, shutting down", name, IDLE_TIMEOUT)
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")
            log.info("[%s] IDLE resume -> WORK", name)
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
            approve = args.get("approve", False)
            new_status = "approved" if approve else "rejected"
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = new_status
            log.info("PROTO shutdown_response sender=%s req_id=%s status=%s", sender, req_id, new_status)
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            log.info("PROTO plan_approval sender=%s req_id=%s plan=%r", sender, req_id, plan_text[:80])
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        log.warning("EXEC unknown tool=%s sender=%s", tool_name, sender)
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
            "description": "Respond to a shutdown request.",
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
            "description": "Submit a plan for lead approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"}
                },
                "required": ["plan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "Signal that you have no more work. Enters idle polling phase.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a task from the task board by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"}
                },
                "required": ["task_id"]
            }
        }
    }
]
    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
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
        log.warning("BASH blocked dangerous command: %r", command[:80])
        return "Error: Dangerous command blocked"
    log.debug("BASH  >>> %r", command[:120])
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        result = out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log.error("BASH  <<< TIMEOUT (120s) cmd=%r", command[:80])
        return "Error: Timeout (120s)"
    log.debug("BASH  <<< elapsed=%.2fs rc=%d out_len=%d", time.perf_counter() - t0, r.returncode, len(result))
    return result
def _run_read(path: str, limit: int = None) -> str:
    log.debug("READ  >>> path=%s limit=%s", path, limit)
    t0 = time.perf_counter()
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        result = "\n".join(lines)[:50000]
    except Exception as e:
        log.error("READ  <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("READ  <<< elapsed=%.2fs lines=%d", time.perf_counter() - t0, len(lines))
    return result
def _run_write(path: str, content: str) -> str:
    log.debug("WRITE >>> path=%s bytes=%d", path, len(content))
    t0 = time.perf_counter()
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(fp, content)
        result = f"Wrote {len(content)} bytes"
    except Exception as e:
        log.error("WRITE <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("WRITE <<< elapsed=%.2fs path=%s", time.perf_counter() - t0, path)
    return result
def _run_edit(path: str, old_text: str, new_text: str) -> str:
    log.debug("EDIT  >>> path=%s old=%r new=%r", path, old_text[:40], new_text[:40])
    t0 = time.perf_counter()
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            log.warning("EDIT  <<< text not found path=%s old=%r", path, old_text[:40])
            return f"Error: Text not found in {path}"
        _atomic_write(fp, c.replace(old_text, new_text, 1))
        result = f"Edited {path}"
    except Exception as e:
        log.error("EDIT  <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("EDIT  <<< elapsed=%.2fs path=%s", time.perf_counter() - t0, path)
    return result
# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    log.info("PROTO shutdown_request req_id=%s target=%s", req_id, teammate)
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"
def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
        if not req:
            log.warning("PROTO plan_review unknown request_id=%s", request_id)
            return f"Error: Unknown plan request_id '{request_id}'"
        req["status"] = "approved" if approve else "rejected"
    log.info("PROTO plan_review req_id=%s approve=%s from=%s", request_id, approve, req["from"])
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"
def _check_shutdown_status(request_id: str) -> str:
    log.debug("PROTO check_shutdown_status req_id=%s", request_id)
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))
# -- Lead tool dispatch (14 tools) --
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
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
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
            "description": "Spawn an autonomous teammate.",
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
                    "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}
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
            "description": "Request a teammate to shut down.",
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
            "name": "shutdown_response",
            "description": "Check shutdown request status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"}
                },
                "required": ["request_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Approve or reject a teammate's plan.",
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
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "Enter idle state (for lead -- rarely used).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a task from the board by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"}
                },
                "required": ["task_id"]
            }
        }
    }
]
_agent_round = 0

def agent_loop(messages: list):
    global _agent_round
    while True:
        _agent_round += 1
        inbox = BUS.read_inbox("lead")
        if inbox:
            log.info("LEAD inbox injected count=%d at round=%d", len(inbox), _agent_round)
            msg_in = {
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            }
            messages.append(msg_in)
            _log_msg_append(messages)
            msg_ack = {
                "role": "assistant",
                "content": "Noted inbox messages.",
            }
            messages.append(msg_ack)
            _log_msg_append(messages)
        log.debug("LEAD Round %d msgs=%d", _agent_round, len(messages))
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        elapsed = time.perf_counter() - t0
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        log.debug("LEAD LLM <<< elapsed=%.2fs stop=%s tool_calls=%d",
                  elapsed, response.choices[0].finish_reason, len(tool_calls))

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
        _log_msg_append(messages)

        if not tool_calls:
            log.info("LEAD done (no tool_calls) round=%d", _agent_round)
            return message.content

        for tool_call in tool_calls:
            assert tool_call.id is not None
            args = json.loads(tool_call.function.arguments or "{}")
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            log.debug("LEAD TOOL >>> %s args=%s", tool_call.function.name, str(args)[:120])
            t1 = time.perf_counter()
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            except Exception as e:
                log.error("LEAD TOOL <<< %s EXCEPTION: %s", tool_call.function.name, e)
                output = f"Error: {e}"
            log.debug("LEAD TOOL <<< %s elapsed=%.2fs out=%r",
                      tool_call.function.name, time.perf_counter() - t1, str(output)[:80])
            print(f"> {tool_call.function.name}: {output[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(output)
            })
            _log_msg_append(messages)
  
if __name__ == "__main__":
    log.info("Agent started model=%s workdir=%s", MODEL, WORKDIR)
    history = [
    {"role": "system", "content": SYSTEM}
]
    _log_msg_append(history)
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
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
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        log.info("USER >>> %r", query[:120])
        history.append({"role": "user", "content": query})
        _log_msg_append(history)
        response_content = agent_loop(history)
        log.info("AGENT <<< %r", str(response_content)[:120])
        print(response_content)
    log.info("Shutting down teammate threads...")
    TEAM.shutdown(timeout=5)
    log.info("Agent exited")
