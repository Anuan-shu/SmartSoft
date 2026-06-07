from __future__ import annotations

import re

from utils.docker_env import DockerEnv
from utils.models import ModelClient
from utils.patches import extract_patch
from utils.tasks import TaskContext


SYSTEM_PROMPT = """You are an autonomous SWE-bench CLI agent running inside a repository container.
Think through shell commands, inspect code, edit files, and validate with tests.

Return EXACTLY ONE of the following formats per turn:
1) For taking an action:
<command>
ONE SINGLE SHELL COMMAND
</command>

2) When you are done:
<final_patch>
UNIFIED DIFF PATCH
</final_patch>

Rules:
- Use non-interactive shell commands only.
- Prefer focused inspections (`ls`, `find`, `grep`, `sed`, `python -m pytest <target>`).
- Work in `/testbed`; do not clone repositories or install large new dependencies.
- First inspect the relevant source and tests, then make the smallest correct code change.
- Use `git diff` to inspect your final changes before returning a patch.
- Never use destructive system-level commands (reboot, shutdown, mkfs, etc.).
- If you edit files, produce the final answer as a unified diff patch.
- Do not output markdown fences or any extra text outside the required tags.
"""

FINAL_PATCH_PROMPT = """You have reached the finalization step.
Return ONLY:
<final_patch>
...unified diff patch...
</final_patch>
No extra text.
"""

MAX_STEPS = 25
MAX_OBS_CHARS = 4000
FORBIDDEN_COMMAND_PATTERNS = (
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "mkfs",
    "fdisk",
    "dd if=",
    "rm -rf /",
    "rm -fr /",
    "git reset --hard",
    "git clean -fd",
    ":(){:|:&};:",
)


def _truncate(text: str, max_chars: int = MAX_OBS_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[:max_chars]}... [truncated]"


def _extract_tagged(text: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _is_forbidden_command(command: str) -> bool:
    lowered = command.lower()
    return any(token in lowered for token in FORBIDDEN_COMMAND_PATTERNS)


def _command_timeout(command: str) -> int:
    lowered = command.lower()
    if any(token in lowered for token in ("pytest", "tox", "unittest", "runtests.py")):
        return 300
    if any(token in lowered for token in ("pip install", "conda install", "apt-get", "npm install")):
        return 300
    return 120


def _current_diff(env: DockerEnv) -> str:
    result = env.run("git diff --no-ext-diff", timeout=60)
    if result.exit_code != 0:
        return ""
    return extract_patch(result.stdout)


def _parse_model_action(content: str) -> tuple[str, str]:
    final_patch = _extract_tagged(content, "final_patch")
    if final_patch:
        return "final_patch", final_patch

    command = _extract_tagged(content, "command")
    if command:
        return "command", command

    # Backward-compatible fallback: if model emits patch directly.
    patch = extract_patch(content)
    if any(patch.startswith(starter) for starter in ("diff --git", "--- ", "Index: ", "*** Begin Patch")):
        return "final_patch", patch
    return "invalid", content.strip()


def _build_task_prompt(
    context: TaskContext,
    history: list[str],
    steps_remaining: int,
    final_only: bool = False,
) -> str:
    history_block = "\n\n".join(history[-8:]).strip() if history else "(none)"
    instruction = (
        "Decide the next best single CLI command."
        if not final_only
        else "Now stop acting and produce the final patch."
    )
    hints = context.hints_text.strip() or "(none)"
    return f"""Problem statement:
{context.problem_statement}

Task metadata:
- instance_id: {context.instance_id}
- repo: {context.repo}
- base_commit: {context.base_commit}
- version: {context.version or "(unknown)"}
- hints: {hints}
- fail_to_pass tests: {context.fail_to_pass}
- pass_to_pass tests: {context.pass_to_pass}

Interaction history:
{history_block}

Instruction:
{instruction}

Steps remaining: {steps_remaining}"""


def _format_observation(exit_code: int, stdout: str, stderr: str) -> str:
    out = _truncate(stdout)
    err = _truncate(stderr)
    return (
        f"exit_code: {exit_code}\n"
        f"stdout:\n{out or '(empty)'}\n"
        f"stderr:\n{err or '(empty)'}"
    )


def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
    history: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        steps_remaining = MAX_STEPS - step + 1
        user_prompt = _build_task_prompt(
            context,
            history,
            steps_remaining=steps_remaining,
            final_only=False,
        )
        response = model.generate(SYSTEM_PROMPT, user_prompt, temperature=0.2)
        action_type, payload = _parse_model_action(response.content)

        if action_type == "final_patch":
            diff = _current_diff(env)
            if diff:
                return diff
            return extract_patch(payload)

        if action_type != "command":
            history.append(
                f"[step {step} invalid]\n"
                f"Model output did not match required tags.\n"
                f"raw:\n{_truncate(response.content)}"
            )
            continue

        command = payload.strip()
        if not command:
            history.append(f"[step {step} invalid]\nEmpty command.")
            continue

        if _is_forbidden_command(command):
            history.append(
                f"[step {step} blocked]\n"
                f"command: {command}\n"
                "reason: blocked by safety policy"
            )
            continue

        result = env.run(command, timeout=_command_timeout(command))
        observation = _format_observation(result.exit_code, result.stdout, result.stderr)
        history.append(f"[step {step} command]\n{command}\n[step {step} observation]\n{observation}")

    diff = _current_diff(env)
    if diff:
        return diff

    # Finalization pass: force model to output only final patch.
    final_prompt = _build_task_prompt(context, history, steps_remaining=0, final_only=True)
    final_response = model.generate(
        SYSTEM_PROMPT + "\n\n" + FINAL_PATCH_PROMPT,
        final_prompt,
        temperature=0.0,
    )
    final_type, final_payload = _parse_model_action(final_response.content)
    if final_type == "final_patch":
        return extract_patch(final_payload)

    # Last fallback: if model already modified files but failed format, emit git diff.
    return _current_diff(env)
