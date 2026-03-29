#!/usr/bin/env python3
"""
s12_worktree_task_isolation.py - Worktree + Task Isolation
Directory-level isolation for parallel task execution.
Tasks are the control plane and worktrees are the execution plane.
    .tasks/task_12.json
      {
        "id": 12,
        "subject": "Implement auth refactor",
        "status": "in_progress",
        "worktree": "auth-refactor"
      }
    .worktrees/index.json
      {
        "worktrees": [
          {
            "name": "auth-refactor",
            "path": ".../.worktrees/auth-refactor",
            "branch": "wt/auth-refactor",
            "task_id": 12,
            "status": "active"
          }
        ]
      }
Key insight: "Isolate by directory, coordinate by task ID."
"""
import json
import logging
import os
import re
import subprocess
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
def detect_repo_root(cwd: Path) -> Path | None:
    """Return git repo root if cwd is inside a repo, else None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        root = Path(r.stdout.strip())
        return root if root.exists() else None
    except Exception:
        return None
REPO_ROOT = detect_repo_root(WORKDIR) or WORKDIR
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use task + worktree tools for multi-task work. "
    "For parallel or risky changes: create tasks, allocate worktree lanes, "
    "run commands in those lanes, then choose keep/remove for closeout. "
    "Use worktree_events when you need lifecycle visibility."
)
# -- EventBus: append-only lifecycle events for observability --
class EventBus:
    def __init__(self, event_log_path: Path):
        self.path = event_log_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
    def emit(
        self,
        event: str,
        task: dict | None = None,
        worktree: dict | None = None,
        error: str | None = None,
    ):
        payload = {
            "event": event,
            "ts": time.time(),
            "task": task or {},
            "worktree": worktree or {},
        }
        if error:
            payload["error"] = error
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    def list_recent(self, limit: int = 20) -> str:
        n = max(1, min(int(limit or 20), 200))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        recent = lines[-n:]
        items = []
        for line in recent:
            try:
                items.append(json.loads(line))
            except Exception:
                items.append({"event": "parse_error", "raw": line})
        return json.dumps(items, indent=2)
# -- TaskManager: persistent task board with optional worktree binding --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._max_id() + 1
    def _max_id(self) -> int:
        ids = []
        for f in self.dir.glob("task_*.json"):
            try:
                ids.append(int(f.stem.split("_")[1]))
            except Exception:
                pass
        return max(ids) if ids else 0
    def _path(self, task_id: int) -> Path:
        return self.dir / f"task_{task_id}.json"
    def _load(self, task_id: int) -> dict:
        path = self._path(task_id)
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())
    def _save(self, task: dict):
        self._path(task["id"]).write_text(json.dumps(task, indent=2))
    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": "",
            "worktree": "",
            "blockedBy": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._save(task)
        log.info("TASK create id=%d subject=%r", task["id"], subject[:60])
        self._next_id += 1
        return json.dumps(task, indent=2)
    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2)
    def exists(self, task_id: int) -> bool:
        return self._path(task_id).exists()
    def update(self, task_id: int, status: str = None, owner: str = None) -> str:
        task = self._load(task_id)
        old_status = task["status"]
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
        if owner is not None:
            task["owner"] = owner
        task["updated_at"] = time.time()
        self._save(task)
        log.info("TASK update id=%d status=%s->%s owner=%s",
                 task_id, old_status, task["status"], task.get("owner",""))
        return json.dumps(task, indent=2)
    def bind_worktree(self, task_id: int, worktree: str, owner: str = "") -> str:
        task = self._load(task_id)
        old_status = task["status"]
        task["worktree"] = worktree
        if owner:
            task["owner"] = owner
        if task["status"] == "pending":
            task["status"] = "in_progress"
        task["updated_at"] = time.time()
        self._save(task)
        log.info("TASK bind_worktree id=%d wt=%s status=%s->%s",
                 task_id, worktree, old_status, task["status"])
        return json.dumps(task, indent=2)
    def unbind_worktree(self, task_id: int) -> str:
        task = self._load(task_id)
        old_wt = task.get("worktree", "")
        task["worktree"] = ""
        task["updated_at"] = time.time()
        self._save(task)
        log.info("TASK unbind_worktree id=%d old_wt=%s", task_id, old_wt)
        return json.dumps(task, indent=2)
    def list_all(self) -> str:
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        log.debug("TASK list_all total=%d", len(tasks))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(t["status"], "[?]")
            owner = f" owner={t['owner']}" if t.get("owner") else ""
            wt = f" wt={t['worktree']}" if t.get("worktree") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{owner}{wt}")
        return "\n".join(lines)
TASKS = TaskManager(REPO_ROOT / ".tasks")
EVENTS = EventBus(REPO_ROOT / ".worktrees" / "events.jsonl")
# -- WorktreeManager: create/list/run/remove git worktrees + lifecycle index --
class WorktreeManager:
    def __init__(self, repo_root: Path, tasks: TaskManager, events: EventBus):
        self.repo_root = repo_root
        self.tasks = tasks
        self.events = events
        self.dir = repo_root / ".worktrees"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"worktrees": []}, indent=2))
        self.git_available = self._is_git_repo()
    def _is_git_repo(self) -> bool:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False
    def _run_git(self, args: list[str]) -> str:
        if not self.git_available:
            raise RuntimeError("Not in a git repository. worktree tools require git.")
        r = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).strip()
            raise RuntimeError(msg or f"git {' '.join(args)} failed")
        return (r.stdout + r.stderr).strip() or "(no output)"
    def _load_index(self) -> dict:
        return json.loads(self.index_path.read_text())
    def _save_index(self, data: dict):
        self.index_path.write_text(json.dumps(data, indent=2))
    def _find(self, name: str) -> dict | None:
        idx = self._load_index()
        for wt in idx.get("worktrees", []):
            if wt.get("name") == name:
                return wt
        return None
    def _validate_name(self, name: str):
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", name or ""):
            raise ValueError(
                "Invalid worktree name. Use 1-40 chars: letters, numbers, ., _, -"
            )
    def create(self, name: str, task_id: int = None, base_ref: str = "HEAD") -> str:
        self._validate_name(name)
        if self._find(name):
            raise ValueError(f"Worktree '{name}' already exists in index")
        if task_id is not None and not self.tasks.exists(task_id):
            raise ValueError(f"Task {task_id} not found")
        path = self.dir / name
        branch = f"wt/{name}"
        log.info("WT   create >>> name=%s task_id=%s base_ref=%s", name, task_id, base_ref)
        self.events.emit(
            "worktree.create.before",
            task={"id": task_id} if task_id is not None else {},
            worktree={"name": name, "base_ref": base_ref},
        )
        t0 = time.perf_counter()
        try:
            self._run_git(["worktree", "add", "-b", branch, str(path), base_ref])
            entry = {
                "name": name,
                "path": str(path),
                "branch": branch,
                "task_id": task_id,
                "status": "active",
                "created_at": time.time(),
            }
            idx = self._load_index()
            idx["worktrees"].append(entry)
            self._save_index(idx)
            if task_id is not None:
                self.tasks.bind_worktree(task_id, name)
            self.events.emit(
                "worktree.create.after",
                task={"id": task_id} if task_id is not None else {},
                worktree={
                    "name": name,
                    "path": str(path),
                    "branch": branch,
                    "status": "active",
                },
            )
            log.info("WT   create <<< name=%s branch=%s elapsed=%.2fs", name, branch, time.perf_counter() - t0)
            return json.dumps(entry, indent=2)
        except Exception as e:
            log.error("WT   create FAILED name=%s: %s", name, e)
            self.events.emit(
                "worktree.create.failed",
                task={"id": task_id} if task_id is not None else {},
                worktree={"name": name, "base_ref": base_ref},
                error=str(e),
            )
            raise
    def list_all(self) -> str:
        idx = self._load_index()
        wts = idx.get("worktrees", [])
        if not wts:
            return "No worktrees in index."
        lines = []
        for wt in wts:
            suffix = f" task={wt['task_id']}" if wt.get("task_id") else ""
            lines.append(
                f"[{wt.get('status', 'unknown')}] {wt['name']} -> "
                f"{wt['path']} ({wt.get('branch', '-')}){suffix}"
            )
        return "\n".join(lines)
    def status(self, name: str) -> str:
        log.debug("WT   status name=%s", name)
        wt = self._find(name)
        if not wt:
            log.warning("WT   status unknown name=%s", name)
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            log.warning("WT   status path missing name=%s path=%s", name, path)
            return f"Error: Worktree path missing: {path}"
        r = subprocess.run(
            ["git", "status", "--short", "--branch"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        text = (r.stdout + r.stderr).strip()
        return text or "Clean worktree"
    def run(self, name: str, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
        if any(d in command for d in dangerous):
            log.warning("WT   run blocked dangerous command name=%s cmd=%r", name, command[:80])
            return "Error: Dangerous command blocked"
        wt = self._find(name)
        if not wt:
            log.warning("WT   run unknown worktree name=%s", name)
            return f"Error: Unknown worktree '{name}'"
        path = Path(wt["path"])
        if not path.exists():
            log.warning("WT   run path missing name=%s path=%s", name, path)
            return f"Error: Worktree path missing: {path}"
        log.debug("WT   run >>> name=%s cmd=%r", name, command[:120])
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            out = (r.stdout + r.stderr).strip()
            result = out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            log.error("WT   run TIMEOUT (300s) name=%s cmd=%r", name, command[:80])
            return "Error: Timeout (300s)"
        log.debug("WT   run <<< name=%s elapsed=%.2fs rc=%d out_len=%d",
                  name, time.perf_counter() - t0, r.returncode, len(result))
        return result
    def remove(self, name: str, force: bool = False, complete_task: bool = False) -> str:
        wt = self._find(name)
        if not wt:
            log.warning("WT   remove unknown name=%s", name)
            return f"Error: Unknown worktree '{name}'"
        log.info("WT   remove >>> name=%s force=%s complete_task=%s task_id=%s",
                 name, force, complete_task, wt.get("task_id"))
        self.events.emit(
            "worktree.remove.before",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={"name": name, "path": wt.get("path")},
        )
        t0 = time.perf_counter()
        try:
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(wt["path"])
            self._run_git(args)
            if complete_task and wt.get("task_id") is not None:
                task_id = wt["task_id"]
                before = json.loads(self.tasks.get(task_id))
                self.tasks.update(task_id, status="completed")
                self.tasks.unbind_worktree(task_id)
                log.info("WT   remove task.completed task_id=%d subject=%r",
                         task_id, before.get("subject","")[:60])
                self.events.emit(
                    "task.completed",
                    task={
                        "id": task_id,
                        "subject": before.get("subject", ""),
                        "status": "completed",
                    },
                    worktree={"name": name},
                )
            idx = self._load_index()
            for item in idx.get("worktrees", []):
                if item.get("name") == name:
                    item["status"] = "removed"
                    item["removed_at"] = time.time()
            self._save_index(idx)
            self.events.emit(
                "worktree.remove.after",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path"), "status": "removed"},
            )
            log.info("WT   remove <<< name=%s elapsed=%.2fs", name, time.perf_counter() - t0)
            return f"Removed worktree '{name}'"
        except Exception as e:
            log.error("WT   remove FAILED name=%s: %s", name, e)
            self.events.emit(
                "worktree.remove.failed",
                task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
                worktree={"name": name, "path": wt.get("path")},
                error=str(e),
            )
            raise
    def keep(self, name: str) -> str:
        wt = self._find(name)
        if not wt:
            log.warning("WT   keep unknown name=%s", name)
            return f"Error: Unknown worktree '{name}'"
        idx = self._load_index()
        kept = None
        for item in idx.get("worktrees", []):
            if item.get("name") == name:
                item["status"] = "kept"
                item["kept_at"] = time.time()
                kept = item
        self._save_index(idx)
        log.info("WT   keep name=%s task_id=%s", name, wt.get("task_id"))
        self.events.emit(
            "worktree.keep",
            task={"id": wt.get("task_id")} if wt.get("task_id") is not None else {},
            worktree={
                "name": name,
                "path": wt.get("path"),
                "status": "kept",
            },
        )
        return json.dumps(kept, indent=2) if kept else f"Error: Unknown worktree '{name}'"
WORKTREES = WorktreeManager(REPO_ROOT, TASKS, EVENTS)
# ── Message logging helpers ────────────────────────────────────────────────────
def _msg_summary(msg: dict) -> str:
    role = msg.get("role", "?")
    if role == "system":
        return f"system: {msg.get('content','')[:60]!r}"
    if role == "tool":
        return f"tool[{msg.get('name','?')}]: {str(msg.get('content',''))[:60]!r}"
    if role == "assistant":
        tcs = msg.get("tool_calls", [])
        if tcs:
            names = [tc["function"]["name"] for tc in tcs]
            return f"assistant: tool_calls={names}"
        return f"assistant: {str(msg.get('content',''))[:60]!r}"
    return f"{role}: {str(msg.get('content',''))[:60]!r}"

def _log_msg_append(messages: list):
    log.debug("MSG  +[%d] %s", len(messages), _msg_summary(messages[-1]))

# -- Base tools (kept minimal, same style as previous sessions) --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        log.warning("BASH blocked dangerous command: %r", command[:80])
        return "Error: Dangerous command blocked"
    log.debug("BASH  >>> %r", command[:120])
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        result = out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        log.error("BASH  <<< TIMEOUT (120s) cmd=%r", command[:80])
        return "Error: Timeout (120s)"
    log.debug("BASH  <<< elapsed=%.2fs rc=%d out_len=%d", time.perf_counter() - t0, r.returncode, len(result))
    return result
def run_read(path: str, limit: int = None) -> str:
    log.debug("READ  >>> path=%s limit=%s", path, limit)
    t0 = time.perf_counter()
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        result = "\n".join(lines)[:50000]
    except Exception as e:
        log.error("READ  <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("READ  <<< elapsed=%.2fs lines=%d", time.perf_counter() - t0, len(lines))
    return result
def run_write(path: str, content: str) -> str:
    log.debug("WRITE >>> path=%s bytes=%d", path, len(content))
    t0 = time.perf_counter()
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        result = f"Wrote {len(content)} bytes"
    except Exception as e:
        log.error("WRITE <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("WRITE <<< elapsed=%.2fs path=%s", time.perf_counter() - t0, path)
    return result
def run_edit(path: str, old_text: str, new_text: str) -> str:
    log.debug("EDIT  >>> path=%s old=%r new=%r", path, old_text[:40], new_text[:40])
    t0 = time.perf_counter()
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            log.warning("EDIT  <<< text not found path=%s old=%r", path, old_text[:40])
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        result = f"Edited {path}"
    except Exception as e:
        log.error("EDIT  <<< ERROR path=%s: %s", path, e)
        return f"Error: {e}"
    log.debug("EDIT  <<< elapsed=%.2fs path=%s", time.perf_counter() - t0, path)
    return result
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("owner")),
    "task_bind_worktree": lambda **kw: TASKS.bind_worktree(kw["task_id"], kw["worktree"], kw.get("owner", "")),
    "worktree_create": lambda **kw: WORKTREES.create(kw["name"], kw.get("task_id"), kw.get("base_ref", "HEAD")),
    "worktree_list": lambda **kw: WORKTREES.list_all(),
    "worktree_status": lambda **kw: WORKTREES.status(kw["name"]),
    "worktree_run": lambda **kw: WORKTREES.run(kw["name"], kw["command"]),
    "worktree_keep": lambda **kw: WORKTREES.keep(kw["name"]),
    "worktree_remove": lambda **kw: WORKTREES.remove(kw["name"], kw.get("force", False), kw.get("complete_task", False)),
    "worktree_events": lambda **kw: EVENTS.list_recent(kw.get("limit", 20)),
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the current workspace (blocking).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
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
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
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
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
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
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "Create a new task on the shared task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "List all tasks with status, owner, and worktree binding.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "Get task details by ID.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "Update task status or owner.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "owner": {"type": "string"},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_bind_worktree",
            "description": "Bind a task to a worktree name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "worktree": {"type": "string"},
                    "owner": {"type": "string"},
                },
                "required": ["task_id", "worktree"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_create",
            "description": "Create a git worktree and optionally bind it to a task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "task_id": {"type": "integer"},
                    "base_ref": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_list",
            "description": "List worktrees tracked in .worktrees/index.json.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_status",
            "description": "Show git status for one worktree.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_run",
            "description": "Run a shell command in a named worktree directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["name", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_remove",
            "description": "Remove a worktree and optionally mark its bound task completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "force": {"type": "boolean"},
                    "complete_task": {"type": "boolean"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_keep",
            "description": "Mark a worktree as kept in lifecycle state without removing it.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "worktree_events",
            "description": "List recent worktree/task lifecycle events from .worktrees/events.jsonl.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
        },
    },
]
_agent_round = 0

def agent_loop(messages: list):
    global _agent_round
    while True:
        _agent_round += 1
        log.debug("AGENT Round %d msgs=%d", _agent_round, len(messages))
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        elapsed = time.perf_counter() - t0
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        log.debug("AGENT LLM <<< elapsed=%.2fs stop=%s tool_calls=%d",
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
            log.info("AGENT done (no tool_calls) round=%d", _agent_round)
            return message.content

        for tool_call in tool_calls:
            assert tool_call.id is not None
            args = json.loads(tool_call.function.arguments)
            handler = TOOL_HANDLERS.get(tool_call.function.name)
            log.debug("AGENT TOOL >>> %s args=%s", tool_call.function.name, str(args)[:120])
            t1 = time.perf_counter()
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_call.function.name}"
            except Exception as e:
                log.error("AGENT TOOL <<< %s EXCEPTION: %s", tool_call.function.name, e)
                output = f"Error: {e}"
            log.debug("AGENT TOOL <<< %s elapsed=%.2fs out=%r",
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
    log.info("Agent started model=%s workdir=%s repo_root=%s git=%s",
             MODEL, WORKDIR, REPO_ROOT, WORKTREES.git_available)
    print(f"Repo root for s12: {REPO_ROOT}")
    if not WORKTREES.git_available:
        print("Note: Not in a git repo. worktree_* tools will return errors.")
    history = [{"role": "system", "content": SYSTEM}]
    _log_msg_append(history)
    while True:
        try:
            query = input("\033[36ms12 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        log.info("USER >>> %r", query[:120])
        history.append({"role": "user", "content": query})
        _log_msg_append(history)
        response_content = agent_loop(history)
        log.info("AGENT <<< %r", str(response_content)[:120])
        print(response_content)
    log.info("Agent exited")