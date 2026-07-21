#!/usr/bin/env python3

import os
import subprocess
import json
import ast

from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

missing_env = [name for name in ("ANTHROPIC_API_KEY", "MODEL_ID") if not os.getenv(name)]
if missing_env:
    raise SystemExit(
        "Missing environment variable(s): "
        + ", ".join(missing_env)
        + "\nCreate a .env file next to agent.py. See .env.example."
    )

API_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL") or None
AUTH_TYPE = os.getenv("ANTHROPIC_AUTH_TYPE") or ("bearer" if BASE_URL else "x-api-key")

if AUTH_TYPE == "bearer":
    client = Anthropic(auth_token=API_KEY, base_url=BASE_URL)
elif AUTH_TYPE == "x-api-key":
    client = Anthropic(api_key=API_KEY, base_url=BASE_URL)
else:
    raise SystemExit("ANTHROPIC_AUTH_TYPE must be either 'bearer' or 'x-api-key'.")

WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
SHELL_NAME = "PowerShell" if os.name == "nt" else "bash"
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Use {SHELL_NAME} commands to solve tasks. "
    "Do not inspect secret files such as .env unless the user explicitly asks. "
    "For follow-up questions, answer from known context before using tools. "
    "Use tools when helpful, then answer the user directly."
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go. "
    "For complex sub-problems, use the task tool to spawn a subagent."
)
SUB_SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Use {SHELL_NAME} commands to solve tasks. "
    "Do not inspect secret files such as .env unless the user explicitly asks. "
    "Complete the task you were given, then return a concise summary. "
    "Do not modify files unless the task description asks you to. "
    "Do not delegate further."
)
# ── Tool definition: just bash ────────────────────────────
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "todo_write", "description": "Plan and track task progress. Pass the full updated todo list.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
]


# ── Tool execution ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════════
#  NEW in s04: Hook System (s03 permission logic now via hooks)
# ════════════════════════════════════════════════════════════════════════════════
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}
def register_hook(event: str, callback):
    HOOKS[event].append(callback)
def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # teaching shortcut: block this tool call
            return result
    return None
# s03 permission check logic, now wrapped as a hook
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = [
    "rm ", "> /etc/", "chmod 777",
    "remove-item", "rmi ", " ri ", " ri",
    "del ", "erase ", "rd ", "rmdir ", "clear-content", ".delete()",
    "[system.io.file]::delete", "[system.io.directory]::delete",
]
def permission_hook(block):
    """PreToolUse: s03 checks in hook form."""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", "").lower():
                print(f'\u001b[31m Blocked: {pattern}\u001b[0m')
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", "").lower():
                print(f'\n\u001b[33m ⚠️  Potentially destructive command\u001b[0m')
                print(f"   Tool: {block.name}({block.input})")
                choice = input("   Allow? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f'\n\u001b[33m ⚠️  Writing outside workspace\u001b[0m')
            print(f"   Tool: {block.name}({block.input})")
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None
def log_hook(block):
    """PreToolUse: log every tool call."""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f'\u001b[90m[HOOK] {block.name}({args_preview})\u001b[0m')
    return None
def large_output_hook(block, output):
    """PostToolUse: warn on large output."""
    if len(str(output)) > 100000:
        print(f'\u001b[33m[HOOK] ⚠️ Large output from {block.name}: {len(str(output))} chars\u001b[0m')
    return None
# UserPromptSubmit hook: log user input before it reaches the LLM
def context_inject_hook(query: str):
    print(f'\u001b[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\u001b[0m')
    return None
# Stop hook: print summary when loop is about to exit
def summary_hook(messages: list):
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f'\u001b[90m[HOOK] Stop: session used {tool_count} tool calls\u001b[0m')
    return None
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


def run_bash(command: str) -> str:
    try:
        if os.name == "nt":
            command = (
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                "$OutputEncoding = [System.Text.Encoding]::UTF8; "
                + command
            )
            shell_command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            shell_command = command
        r = subprocess.run(shell_command, shell=not isinstance(shell_command, list), cwd=os.getcwd(),
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        out = redact_secrets(((r.stdout or "") + (r.stderr or "")).strip())
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def is_sensitive_path(path: Path) -> bool:
    sensitive_names = {".env", ".env.local", ".env.production", ".env.development"}
    return path.name.lower() in sensitive_names

def redact_secrets(text: str) -> str:
    if API_KEY:
        text = text.replace(API_KEY, "[REDACTED_API_KEY]")
    return text

def run_read(path: str, limit: int | None = None) -> str:
    try:
        file_path = safe_path(path)
        if is_sensitive_path(file_path):
            return f"Error: Refusing to read sensitive file: {path}"
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"
    
CURRENT_TODOS = []
def _normalize_todos(todos):
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError):
                return None, "Error: todos must be a list or JSON array string"
    if not isinstance(todos, list):
        return None, "Error: todos must be a list"
    for i, t in enumerate(todos):
        if not isinstance(t, dict):
            return None, f"Error: todos[{i}] must be an object"
        if "content" not in t or "status" not in t:
            return None, f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return None, f"Error: todos[{i}] has invalid status '{t['status']}'"
    return todos, None
def run_todo_write(todos: list) -> str:
    global CURRENT_TODOS
    todos, error = _normalize_todos(todos)
    if error:
        return error
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## Current Tasks\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m\u25b6\033[0m", "completed": "\033[32m\u2713\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}

# ═══════════════════════════════════════════════════════════
#  s06: Subagent — fresh messages[], summary only
# ═══════════════════════════════════════════════════════════
SUB_TOOL_NAMES = ("bash", "read_file", "write_file", "edit_file", "glob")
SUB_TOOLS = [tool for tool in TOOLS if tool["name"] in SUB_TOOL_NAMES]
SUB_HANDLERS = {name: TOOL_HANDLERS[name] for name in SUB_TOOL_NAMES}

def extract_text(content) -> str:
    """Extract text from Anthropic message content blocks."""
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)

def spawn_subagent(description: str) -> str:
    """Spawn a subagent with fresh messages[], return summary only."""
    print(f"\n\033[35m[Subagent spawned]\033[0m")
    messages = [{"role": "user", "content": description}]
    hit_limit = True
    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            hit_limit = False
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue
            handler = SUB_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            trigger_hooks("PostToolUse", block, output)
            print(f"  \033[90m[sub] {block.name}: {str(output)[:100]}\033[0m")
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output})
        messages.append({"role": "user", "content": results})

    result = extract_text(messages[-1]["content"])
    if hit_limit:
        result = "Subagent stopped after 30 turns without final answer."
    if not result:
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                result = extract_text(msg["content"])
                if result:
                    break
        if not result:
            result = "Subagent stopped after 30 turns without final answer."
    print(f"\033[35m[Subagent done]\033[0m")
    return result

TOOLS.append({
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
})
TOOL_HANDLERS["task"] = spawn_subagent

# ── The core pattern: a while loop that calls tools until the model stops ──
rounds_since_todo = 0
def agent_loop(messages: list):
    global rounds_since_todo
    while True:
        # s05: nag reminder — inject if model hasn't updated todos for 3 rounds
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return
        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            # s04 change: hook replaces hard-coded check_permission()
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            trigger_hooks("PostToolUse", block, output)  # s04: post hook
            # s05: reset nag counter when todo_write is called
            if block.name == "todo_write":
                rounds_since_todo = 0
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("agent: Tool Use + Subagent")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36magent >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
