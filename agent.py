#!/usr/bin/env python3

import os
import subprocess
import json
import ast
import time
import re
import random

import yaml
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
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
SKILLS_DIR = WORKDIR / "skills"
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"
MODEL = os.environ["MODEL_ID"]
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")
SHELL_NAME = "PowerShell" if os.name == "nt" else "bash"

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from SKILL.md. Returns (meta, body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()

SKILL_REGISTRY: dict[str, dict] = {}
def _scan_skills():
    """Scan skills/ and populate the registry with each SKILL.md."""
    if not SKILLS_DIR.exists():
        return
    for directory in sorted(SKILLS_DIR.iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text(encoding="utf-8", errors="replace")
            meta, _ = _parse_frontmatter(raw)
            name = meta.get("name", directory.name)
            description = meta.get("description", raw.split("\n")[0].lstrip("#").strip())
            SKILL_REGISTRY[name] = {
                "name": name,
                "description": description,
                "content": raw,
            }

_scan_skills()

def list_skills() -> str:
    """List all available skills with their short descriptions."""
    if not SKILL_REGISTRY:
        return "(no skills found)"
    return "\n".join(
        f"- **{skill['name']}**: {skill['description']}"
        for skill in SKILL_REGISTRY.values()
    )

MEMORY_TYPES = {"user", "feedback", "project", "reference"}

def write_memory_file(name: str, mem_type: str, description: str, body: str) -> Path:
    """Write one memory file and refresh the lightweight index."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")
    filename = f"{slug or f'memory-{int(time.time())}'}.md"
    path = MEMORY_DIR / filename
    metadata = yaml.safe_dump(
        {
            "name": name,
            "description": description,
            "type": mem_type if mem_type in MEMORY_TYPES else "user",
        },
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    path.write_text(f"---\n{metadata}\n---\n\n{body}\n", encoding="utf-8")
    _rebuild_memory_index()
    return path

def _rebuild_memory_index() -> None:
    lines = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        if path == MEMORY_INDEX:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", path.stem)
        description = meta.get("description", body.split("\n", 1)[0][:80])
        lines.append(f"- [{name}]({path.name}) - {description}")
    content = "\n".join(lines)
    MEMORY_INDEX.write_text(f"{content}\n" if content else "", encoding="utf-8")

def read_memory_index() -> str:
    if not MEMORY_INDEX.exists():
        return ""
    return MEMORY_INDEX.read_text(encoding="utf-8", errors="replace").strip()

def read_memory_file(filename: str) -> str | None:
    path = MEMORY_DIR / filename
    if not path.is_file() or not path.resolve().is_relative_to(MEMORY_DIR.resolve()):
        return None
    return path.read_text(encoding="utf-8", errors="replace")

def list_memory_files() -> list[dict]:
    memories = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        if path == MEMORY_INDEX:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(raw)
        memories.append({
            "filename": path.name,
            "name": meta.get("name", path.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
        })
    return memories

def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    memories = list_memory_files()
    if not memories:
        return []

    recent_texts = []
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        text = extract_text(message.get("content", ""))
        if text:
            recent_texts.append(text)
        if len(recent_texts) >= 3:
            break
    recent = " ".join(reversed(recent_texts))[:2000]
    if not recent.strip():
        return []

    catalog = "\n".join(
        f"{index}: {memory['name']} - {memory['description']}"
        for index, memory in enumerate(memories)
    )
    prompt = (
        "Given the recent conversation and the memory catalog below, select the "
        "indices of memories that are clearly relevant. Return ONLY a JSON array "
        "of integers, e.g. [0, 3]. If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\nMemory catalog:\n{catalog}"
    )
    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        match = re.search(r"\[.*?\]", extract_text(response.content), re.DOTALL)
        if match:
            indices = json.loads(match.group())
            return [
                memories[index]["filename"]
                for index in indices
                if isinstance(index, int) and 0 <= index < len(memories)
            ][:max_items]
    except Exception:
        pass

    keywords = [word.lower() for word in recent.split() if len(word) > 3]
    return [
        memory["filename"]
        for memory in memories
        if any(
            keyword in f"{memory['name']} {memory['description']}".lower()
            for keyword in keywords
        )
    ][:max_items]

def load_memories(messages: list) -> str:
    selected = select_relevant_memories(messages)
    contents = [read_memory_file(filename) for filename in selected]
    contents = [content for content in contents if content]
    if not contents:
        return ""
    return "\n\n".join(["<relevant_memories>", *contents, "</relevant_memories>"])

def extract_memories(messages: list) -> None:
    dialogue = "\n".join(
        f"{message.get('role', '?')}: {text}"
        for message in messages[-10:]
        if (text := extract_text(message.get("content", ""))).strip()
    )
    if not dialogue:
        return

    existing = list_memory_files()
    existing_descriptions = "\n".join(
        f"- {memory['name']}: {memory['description']}" for memory in existing
    ) or "(none)"
    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier\n"
        "- type: one of user, feedback, project, reference\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing is new or it is already covered, return [].\n\n"
        f"Existing memories:\n{existing_descriptions}\n\nDialogue:\n{dialogue[:4000]}"
    )
    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        match = re.search(r"\[.*\]", extract_text(response.content), re.DOTALL)
        if not match:
            return
        memories = json.loads(match.group())
        count = 0
        for memory in memories:
            description = memory.get("description", "")
            body = memory.get("body", "")
            if description and body:
                write_memory_file(
                    memory.get("name", f"memory-{int(time.time())}"),
                    memory.get("type", "user"),
                    description,
                    body,
                )
                count += 1
        if count:
            print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
    except Exception:
        pass

CONSOLIDATE_THRESHOLD = 10

def consolidate_memories() -> None:
    memories = list_memory_files()
    if len(memories) < CONSOLIDATE_THRESHOLD:
        return

    catalog = "\n\n".join(
        f"## {memory['filename']}\nname: {memory['name']}\n"
        f"description: {memory['description']}\n{memory['body']}"
        for memory in memories
    )
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated or contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )
    try:
        response = client.messages.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
        )
        match = re.search(r"\[.*\]", extract_text(response.content), re.DOTALL)
        consolidated = json.loads(match.group()) if match else None
        if not isinstance(consolidated, list):
            return
        for path in MEMORY_DIR.glob("*.md"):
            if path != MEMORY_INDEX:
                path.unlink()
        for memory in consolidated:
            description = memory.get("description", "")
            body = memory.get("body", "")
            if description and body:
                write_memory_file(
                    memory.get("name", f"memory-{int(time.time())}"),
                    memory.get("type", "user"),
                    description,
                    body,
                )
        _rebuild_memory_index()
        print(
            f"\n\033[33m[Memory: consolidated {len(memories)} -> "
            f"{len(consolidated)} memories]\033[0m"
        )
    except Exception:
        pass

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Use tools when helpful, then answer the user directly.",
    "workspace": (
        "Working directory: {workspace}. Use {shell_name} commands to solve tasks. "
        "Do not inspect secret files such as .env unless the user explicitly asks."
    ),
    "tools": "Available tools: {enabled_tools}.",
    "workflow": (
        "For follow-up questions, answer from known context before using tools. "
        "Before starting any multi-step task, use todo_write to plan your steps. "
        "Update status as you go. For complex sub-problems, use the task tool to spawn a subagent."
    ),
    "skills": (
        "Skills available:\n{skills}\n"
        "Use load_skill to get full details when needed."
    ),
    "memory": (
        "Relevant memories are provided in the system context when needed. "
        "Respect user preferences from memory."
    ),
}

def assemble_system_prompt(context: dict) -> str:
    """Select and join stable prompt sections from the current runtime context."""
    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["workspace"].format(
            workspace=context["workspace"], shell_name=SHELL_NAME
        ),
        PROMPT_SECTIONS["tools"].format(
            enabled_tools=", ".join(context["enabled_tools"])
        ),
        PROMPT_SECTIONS["workflow"],
        PROMPT_SECTIONS["skills"].format(skills=context["skills"]),
    ]
    memory_index = context.get("memory_index", "")
    if memory_index:
        sections.append(f"Memories available:\n{memory_index}")
    memories = context.get("memories", "")
    if memories:
        sections.append(memories)
    sections.append(PROMPT_SECTIONS["memory"])
    return "\n\n".join(sections)

_last_context_key = None
_last_prompt = None

def get_system_prompt(context: dict) -> str:
    """Return a cached prompt when the runtime context has not changed."""
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt is not None:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    loaded = ["identity", "workspace", "tools", "workflow", "skills"]
    if context.get("memory_index"):
        loaded.append("memory index")
    if context.get("memories"):
        loaded.append("relevant memories")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt

def update_context(context: dict, messages: list) -> dict:
    """Refresh prompt inputs from tools, workspace, skills, and memory state."""
    memory_index = read_memory_index()
    memory_changed = memory_index != context.get("memory_index")
    if memory_changed or "memories" not in context:
        memories = load_memories(messages)
    else:
        memories = context["memories"]
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "skills": list_skills(),
        "memory_index": memory_index,
        "memories": memories,
    }

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

def load_skill(name: str) -> str:
    """Load full skill content by registered name."""
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        return f"Skill not found: {name}"
    return skill["content"]

TOOLS.append({
    "name": "task",
    "description": "Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
    "input_schema": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]},
})
TOOL_HANDLERS["task"] = spawn_subagent
TOOLS.append({
    "name": "load_skill",
    "description": "Load the full content of a skill by name.",
    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
})
TOOL_HANDLERS["load_skill"] = load_skill

# ── Context compaction ────────────────────────────────────
CONTEXT_LIMIT = 50_000
KEEP_RECENT = 3
PERSIST_THRESHOLD = 30_000
DEFAULT_MAX_TOKENS = 8_000
ESCALATED_MAX_TOKENS = 64_000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_CONSECUTIVE_529 = 3
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly - "
    "no apology, no recap. Pick up mid-thought."
)

class RecoveryState:
    """Track recovery attempts across one agent loop."""
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = MODEL

def retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Return an exponential retry delay with jitter."""
    if retry_after is not None:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32_000) / 1000
    return base + random.uniform(0, base * 0.25)

def get_retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return max(0.0, float(headers.get("retry-after")))
    except (TypeError, ValueError):
        return None

def with_retry(fn, state: RecoveryState):
    """Retry rate-limit and overload errors; re-raise other failures."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as error:
            name = type(error).__name__.lower()
            message = str(error).lower()
            if "ratelimit" in name or "429" in message:
                delay = retry_delay(attempt, get_retry_after(error))
                print(
                    f"  \033[33m[429 rate limit] retry {attempt + 1}/{MAX_RETRIES}, "
                    f"wait {delay:.1f}s\033[0m"
                )
            elif "overloaded" in name or "529" in message or "overloaded" in message:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL:
                        state.current_model = FALLBACK_MODEL
                        print(
                            f"  \033[31m[529 x{MAX_CONSECUTIVE_529}] "
                            f"switching to {FALLBACK_MODEL}\033[0m"
                        )
                    else:
                        print(
                            f"  \033[31m[529 x{MAX_CONSECUTIVE_529}] "
                            "no FALLBACK_MODEL_ID configured, continuing retry\033[0m"
                        )
                    state.consecutive_529 = 0
                delay = retry_delay(attempt, get_retry_after(error))
                print(
                    f"  \033[33m[529 overloaded] retry {attempt + 1}/{MAX_RETRIES}, "
                    f"wait {delay:.1f}s\033[0m"
                )
            else:
                raise
            if attempt + 1 < MAX_RETRIES:
                time.sleep(delay)
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")

def is_prompt_too_long_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        ("prompt" in message and "long" in message)
        or "prompt_is_too_long" in message
        or "context_length_exceeded" in message
        or "max_context_window" in message
        or "too many tokens" in message
    )

def estimate_size(messages: list) -> int:
    return len(str(messages))

def _block_type(block) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

def _message_has_tool_use(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "assistant"
        and isinstance(content, list)
        and any(_block_type(block) == "tool_use" for block in content)
    )

def _is_tool_result_message(message: dict) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    )

def snip_compact(messages: list, max_messages: int = 50) -> list:
    if len(messages) <= max_messages:
        return messages

    head_end = 3
    tail_start = len(messages) - (max_messages - 3)
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    if (
        tail_start > 0
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end
    return messages[:head_end] + [
        {"role": "user", "content": f"[snipped {snipped} messages]"}
    ] + messages[tail_start:]

def collect_tool_results(messages: list) -> list[dict]:
    results = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        results.extend(
            block for block in content
            if isinstance(block, dict) and block.get("type") == "tool_result"
        )
    return results

def micro_compact(messages: list) -> list:
    tool_results = collect_tool_results(messages)
    for block in tool_results[:-KEEP_RECENT]:
        if len(str(block.get("content", ""))) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
    return messages

def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not path.exists():
        path.write_text(output, encoding="utf-8")
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

def tool_result_budget(messages: list, max_bytes: int = 200_000) -> list:
    if not messages:
        return messages
    content = messages[-1].get("content")
    if messages[-1].get("role") != "user" or not isinstance(content, list):
        return messages

    blocks = [block for block in content if isinstance(block, dict) and block.get("type") == "tool_result"]
    total = sum(len(str(block.get("content", ""))) for block in blocks)
    for block in sorted(blocks, key=lambda item: len(str(item.get("content", ""))), reverse=True):
        if total <= max_bytes:
            break
        output = str(block.get("content", ""))
        if len(output) <= PERSIST_THRESHOLD:
            continue
        block["content"] = persist_large_output(block.get("tool_use_id", "unknown"), output)
        total = sum(len(str(item.get("content", ""))) for item in blocks)
    return messages

def write_transcript(messages: list) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w", encoding="utf-8") as transcript:
        for message in messages:
            transcript.write(json.dumps(message, default=str, ensure_ascii=False) + "\n")
    return path

def summarize_history(messages: list) -> str:
    conversation = json.dumps(messages, default=str, ensure_ascii=False)[:80_000]
    prompt = (
        "Summarize this coding-agent conversation so work can continue.\n"
        "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
        "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n"
        + conversation
    )
    response = client.messages.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2000
    )
    return extract_text(response.content).strip() or "(empty summary)"

def compact_history(messages: list) -> list:
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    return [{"role": "user", "content": f"[Compacted]\n\n{summarize_history(messages)}"}]

def reactive_compact(messages: list) -> list:
    write_transcript(messages)
    tail_start = max(0, len(messages) - 5)
    if (
        tail_start > 0
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    return [
        {"role": "user", "content": f"[Reactive compact]\n\n{summarize_history(messages)}"},
        *messages[tail_start:],
    ]

TOOLS.append({
    "name": "compact",
    "description": "Summarize earlier conversation to free context space.",
    "input_schema": {"type": "object", "properties": {"focus": {"type": "string"}}},
})

# ── The core pattern: a while loop that calls tools until the model stops ──
rounds_since_todo = 0
def agent_loop(messages: list, context: dict | None = None):
    global rounds_since_todo
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS
    context = update_context(context or {}, messages)
    system = get_system_prompt(context)
    memory_source = [
        {"role": message.get("role", "?"), "content": extract_text(message.get("content", ""))}
        for message in messages[-9:]
    ]
    while True:
        # s05: nag reminder — inject if model hasn't updated todos for 3 rounds
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0
        messages[:] = tool_result_budget(messages)
        messages[:] = snip_compact(messages)
        messages[:] = micro_compact(messages)
        if estimate_size(messages) > CONTEXT_LIMIT:
            print("[auto compact]")
            messages[:] = compact_history(messages)
        try:
            response = with_retry(
                lambda: client.messages.create(
                    model=state.current_model,
                    system=system,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=max_tokens,
                ),
                state,
            )
        except Exception as error:
            if is_prompt_too_long_error(error) and not state.has_attempted_reactive_compact:
                print("[reactive compact]")
                messages[:] = reactive_compact(messages)
                state.has_attempted_reactive_compact = True
                continue
            if is_prompt_too_long_error(error):
                text = "[Error] Context too large, cannot continue."
            else:
                name = type(error).__name__
                text = f"[Error] {name}: {str(error)[:200]}"
            print(f"  \033[31m[unrecoverable] {text}\033[0m")
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": text}
            ]})
            return
        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(
                    f"  \033[33m[max_tokens] escalating "
                    f"{DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m"
                )
                continue
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(
                    f"  \033[33m[max_tokens] continuation "
                    f"{state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m"
                )
                continue
            print("  \033[31m[max_tokens] recovery limit reached\033[0m")
            return
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            memory_source.append({
                "role": "assistant",
                "content": extract_text(response.content),
            })
            extract_memories(memory_source)
            consolidate_memories()
            return
        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "compact":
                messages[:] = compact_history(messages)
                break
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
        else:
            messages.append({"role": "user", "content": results})
            context = update_context(context, messages)
            system = get_system_prompt(context)
            continue
        context = update_context(context, messages)
        system = get_system_prompt(context)

# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    print("agent: Tool Use + Subagent + Memory")
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
