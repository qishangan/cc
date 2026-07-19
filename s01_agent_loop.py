#!/usr/bin/env python3

import os
import subprocess

from anthropic import Anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

missing_env = [name for name in ("ANTHROPIC_API_KEY", "MODEL_ID") if not os.getenv(name)]
if missing_env:
    raise SystemExit(
        "Missing environment variable(s): "
        + ", ".join(missing_env)
        + "\nCreate a .env file next to s01_agent_loop.py. See .env.example."
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
]


# ── Tool execution ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════
#  NEW in s03: Three-Gate Permission Pipeline
# ═══════════════════════════════════════════════════════════
# Gate 1: Hard deny list — always forbidden
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda",
    "format-volume", "clear-disk", "remove-partition",
]
def check_deny_list(command: str) -> str | None:
    command = command.lower()
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None

# Gate 2: Rule matching — context-dependent checks
def is_workspace_escape(args: dict) -> bool:
    return not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR)

def command_text(args: dict) -> str:
    return args.get("command", "").lower()

def is_destructive_shell_command(args: dict) -> bool:
    command = command_text(args)
    destructive_patterns = [
        "rm ", "rm\t", "rm\n", "remove-item", "rmi ", " ri ", "ri ",
        "del ", "erase ", "rd ", "rmdir ", "clear-content", ".delete()",
        "[system.io.file]::delete", "[system.io.directory]::delete",
    ]
    return any(pattern in command for pattern in destructive_patterns)

def touches_windows_temp(args: dict) -> bool:
    command = command_text(args).replace("\\", "/")
    temp = (os.environ.get("TEMP") or os.environ.get("TMP") or "").lower().replace("\\", "/")
    temp_markers = [
        "$env:temp", "$env:tmp", "%temp%", "%tmp%", "$temp", "$tmp",
        "[io.path]::gettemppath", "appdata/local/temp", "/tmp",
    ]
    if temp:
        temp_markers.append(temp)
    return any(marker in command for marker in temp_markers)

PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     "check": is_workspace_escape,
     "message": "Writing outside workspace"},
    {"tools": ["bash"],
     "check": lambda args: any(kw in command_text(args) for kw in ["> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
    {"tools": ["bash"],
     "check": lambda args: is_destructive_shell_command(args) and touches_windows_temp(args),
     "message": "Deleting files from the Windows temp directory"},
    {"tools": ["bash"],
     "check": is_destructive_shell_command,
     "message": "Potentially destructive shell command"},
]
def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

# Gate 3: User approval — wait for confirmation after rule match
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"

# Pipeline: all three gates chained
def check_permission(block) -> bool:
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
    return True

def run_bash(command: str) -> str:
    reason = check_deny_list(command)
    if reason:
        return f"Error: {reason}"
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
    
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

# ── The core pattern: a while loop that calls tools until the model stops ──
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m> {block.name}\033[0m")
                if not check_permission(block):
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Permission denied."})
                    continue
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                print(redact_secrets(str(output))[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("s02: Tool Use — 在 s01 基础上加了 4 个工具")
    print("输入问题，回车发送。输入 q 退出。\n")
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # Print the model's final text response
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
