from __future__ import annotations

import base64
import re

from utils.docker_env import DockerEnv
from utils.models import ModelClient
from utils.patches import extract_patch
from utils.tasks import TaskContext


SYSTEM_PROMPT = """You are an autonomous SWE-bench CLI agent running inside a repository container at `/testbed`.
Your goal is to make the FAIL_TO_PASS tests pass while keeping the PASS_TO_PASS tests green,
by editing the real source files in the container and validating with tests.

Return EXACTLY ONE of the following formats per turn (nothing else):
1) To run a single shell command:
<command>
ONE SINGLE SHELL COMMAND
</command>

2) When the fix is applied and verified:
<final_patch>
UNIFIED DIFF PATCH
</final_patch>

Recommended workflow:
- Step 1: reproduce the failure by running the FAIL_TO_PASS test(s), e.g.
  `cd /testbed && python -m pytest <test_id> -x` so you see the real error.
- Step 2: locate the relevant source with `grep -rn` / `sed -n 'A,Bp' file` and read the exact code.
- Step 3: APPLY the fix DIRECTLY to the file in the container. You must actually modify files,
  not just read them. Edit using one self-contained command, for example a Python heredoc:
  ```
  python - <<'PY'
  import pathlib
  p = pathlib.Path("astropy/io/ascii/rst.py")
  text = p.read_text()
  text = text.replace("OLD EXACT SNIPPET", "NEW SNIPPET")
  p.write_text(text)
  PY
  ```
  or a heredoc `git apply <<'EOF' ... EOF`. Prefer replacing exact, unique substrings.
- Step 4: re-run the FAIL_TO_PASS and a few PASS_TO_PASS tests to confirm the fix works.
- Step 5: run `git diff --no-ext-diff` to confirm your edits, then return `<final_patch>` with that diff.

Rules:
- Use only non-interactive shell commands; one command per turn.
- Make the SMALLEST correct change. Change production source files only; do not edit test files
  unless the task explicitly requires it.
- The returned patch is graded by re-applying it to a clean checkout, so it MUST reflect edits you
  actually made in `/testbed`. The framework will re-derive the patch from `git diff` for you, so
  always make your edits land on disk rather than inventing a diff from memory.
- Do not clone repositories or install large new dependencies; assume the environment is ready.
- Never use destructive system-level commands (reboot, shutdown, mkfs, rm -rf /, git reset --hard, etc.).
- Do not output markdown fences or any extra text outside the required tags.
"""

FINAL_PATCH_PROMPT = """You have reached the finalization step.
Return ONLY:
<final_patch>
...unified diff patch...
</final_patch>
No extra text.
The patch must be produced from `git diff --no-ext-diff` whenever possible.
Do not include repeated hunks, truncated lines, or whitespace-only changes.
"""

MAX_STEPS = 25
MAX_OBS_CHARS = 4000
MAX_MODEL_PATCH_LINES = 350
HUNK_HEADER_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@(?P<suffix>.*)$")
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
    return _normalize_patch(result.stdout)


def _format_range(start: str, count: int) -> str:
    return start if count == 1 else f"{start},{count}"


def _repair_hunk_headers(patch: str) -> str:
    lines = patch.splitlines()
    repaired: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        match = HUNK_HEADER_RE.match(line)
        if not match:
            repaired.append(line)
            index += 1
            continue

        header_index = len(repaired)
        repaired.append(line)
        index += 1
        old_count = 0
        new_count = 0

        while index < len(lines):
            body_line = lines[index]
            if body_line.startswith("diff --git ") or HUNK_HEADER_RE.match(body_line):
                break

            # Models sometimes emit an empty context line without the leading
            # space required by unified diff. Preserve intent and keep counts valid.
            if body_line == "":
                body_line = " "

            if body_line.startswith(" "):
                old_count += 1
                new_count += 1
            elif body_line.startswith("-"):
                old_count += 1
            elif body_line.startswith("+"):
                new_count += 1
            elif not body_line.startswith("\\"):
                body_line = f" {body_line}"
                old_count += 1
                new_count += 1

            repaired.append(body_line)
            index += 1

        old_range = _format_range(match.group("old_start"), old_count)
        new_range = _format_range(match.group("new_start"), new_count)
        repaired[header_index] = f"@@ -{old_range} +{new_range} @@{match.group('suffix')}"

    return "\n".join(repaired)


def _normalize_patch(patch: str) -> str:
    cleaned = extract_patch(patch)
    if not cleaned:
        return ""
    cleaned = _repair_hunk_headers(cleaned)
    return cleaned if cleaned.endswith("\n") else f"{cleaned}\n"


def _patch_quality_issue(patch: str) -> str:
    cleaned = _normalize_patch(patch)
    if not cleaned:
        return "empty patch"

    lines = cleaned.splitlines()
    if len(lines) > MAX_MODEL_PATCH_LINES:
        return f"patch is too large ({len(lines)} lines); likely repeated or runaway output"

    hunk_bodies: dict[tuple[str, ...], int] = {}
    index = 0
    while index < len(lines):
        if not HUNK_HEADER_RE.match(lines[index]):
            index += 1
            continue
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.startswith("diff --git ") or HUNK_HEADER_RE.match(line):
                break
            body.append(line.rstrip())
            index += 1
        signature = tuple(body)
        if signature:
            hunk_bodies[signature] = hunk_bodies.get(signature, 0) + 1
            if hunk_bodies[signature] > 1:
                return "patch contains repeated hunks"

    meaningful_changes = [
        line
        for line in lines
        if line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
        and line[1:].strip()
    ]
    if not meaningful_changes:
        return "patch has no meaningful changed lines"

    last_nonempty = next((line for line in reversed(lines) if line.strip()), "")
    if last_nonempty and not last_nonempty.startswith((" ", "+", "-", "\\", "@@", "diff --git", "index ", "---", "+++")):
        return f"patch appears truncated near: {last_nonempty[:80]}"

    return ""


PATCH_FILE = "/tmp/swe_agent_patch.diff"
APPLY_STRATEGIES = (
    f"git apply --whitespace=nowarn {PATCH_FILE}",
    f"git apply --3way --whitespace=nowarn {PATCH_FILE}",
    f"git apply -C1 --whitespace=nowarn {PATCH_FILE}",
    f"patch -p1 --fuzz=3 --no-backup-if-mismatch -i {PATCH_FILE}",
)


def _write_patch_file(env: DockerEnv, patch: str) -> None:
    encoded = base64.b64encode(patch.encode("utf-8")).decode("ascii")
    env.run(
        "python -c "
        f"\"import base64, pathlib; pathlib.Path('{PATCH_FILE}').write_bytes(base64.b64decode('{encoded}'))\"",
        timeout=60,
    )


def _check_patch(env: DockerEnv, patch: str) -> tuple[bool, str]:
    cleaned = _normalize_patch(patch)
    if not cleaned:
        return False, "empty patch"
    quality_issue = _patch_quality_issue(cleaned)
    if quality_issue:
        return False, quality_issue
    _write_patch_file(env, cleaned)
    result = env.run(f"git apply --check {PATCH_FILE}", timeout=60)
    if result.exit_code == 0:
        return True, "git apply --check succeeded"

    dry_run = env.run(f"patch --dry-run --batch --fuzz=5 -p1 -i {PATCH_FILE}", timeout=60)
    if dry_run.exit_code == 0:
        return True, "patch dry-run succeeded"

    detail = dry_run.stderr or dry_run.stdout or result.stderr or result.stdout
    return False, _truncate(detail, max_chars=1000)


def _apply_patch_to_container(env: DockerEnv, patch: str) -> tuple[bool, str]:
    """Materialize the model patch onto disk so the final diff matches the repo state."""
    cleaned = _normalize_patch(patch)
    if not cleaned:
        return False, "empty patch"
    quality_issue = _patch_quality_issue(cleaned)
    if quality_issue:
        return False, quality_issue
    _write_patch_file(env, cleaned)
    last_detail = ""
    for command in APPLY_STRATEGIES:
        result = env.run(command, timeout=120)
        if result.exit_code == 0:
            return True, command
        last_detail = result.stderr or result.stdout or last_detail
    return False, _truncate(last_detail, max_chars=1000)


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


def _finalize_model_patch(env: DockerEnv, payload: str) -> tuple[str, str]:
    """Return (patch, reason). Prefer the real git diff after applying edits to disk."""
    diff = _current_diff(env)
    if diff:
        return diff, "git diff from container edits"

    applied, detail = _apply_patch_to_container(env, payload)
    if applied:
        real_diff = _current_diff(env)
        if real_diff:
            return real_diff, f"applied model patch then re-derived git diff ({detail})"
        normalized = _normalize_patch(payload)
        if normalized:
            return normalized, "applied model patch (git diff empty, returning normalized patch)"
    return "", detail or "model patch could not be applied to the container"


def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
    history: list[str] = []
    executed_commands: set[str] = set()

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
            patch, reason = _finalize_model_patch(env, payload)
            if patch:
                return patch
            history.append(
                f"[step {step} invalid]\n"
                "Model returned a final patch, but no files were edited in the container "
                "and the supplied patch does not apply cleanly.\n"
                f"apply failure:\n{reason}\n"
                "Continue by running commands that EDIT files in /testbed (e.g. a python heredoc that "
                "rewrites the source), then verify with tests."
            )
            continue

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

        repeated_note = ""
        if command in executed_commands:
            repeated_note = (
                "\nnote: this exact command was already run earlier with the same result. "
                "Stop re-inspecting and EDIT the source file now, then run the tests."
            )
        executed_commands.add(command)

        result = env.run(command, timeout=_command_timeout(command))
        observation = _format_observation(result.exit_code, result.stdout, result.stderr)
        history.append(
            f"[step {step} command]\n{command}\n[step {step} observation]\n{observation}{repeated_note}"
        )

    # Out of steps: prefer real edits already on disk.
    diff = _current_diff(env)
    if diff:
        return diff

    # Finalization pass: force model to output only the final patch.
    final_prompt = _build_task_prompt(context, history, steps_remaining=0, final_only=True)
    final_response = model.generate(
        SYSTEM_PROMPT + "\n\n" + FINAL_PATCH_PROMPT,
        final_prompt,
        temperature=0.0,
    )
    final_type, final_payload = _parse_model_action(final_response.content)
    if final_type == "final_patch":
        patch, _reason = _finalize_model_patch(env, final_payload)
        if patch:
            return patch

    # Last fallback: if model already modified files but failed format, emit git diff.
    return _current_diff(env)
