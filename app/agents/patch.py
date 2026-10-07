"""
Patch protocol.

An LLM cannot touch the filesystem. It can only *propose* text. This module is
the narrow, deterministic bridge between the two: it parses a strict,
easy-to-produce envelope out of the model's reply and applies it to one
isolated worktree.

The contract the model is asked to follow:

    <file path="src/api/routes.py">
    ...full file contents...
    </file>

    <delete path="src/old.py"/>

    <note>Explain what you did and what still needs to happen.</note>

Everything else in the reply is ignored. Anything that tries to escape the
workspace is rejected. This keeps "the LLM decides what code to write" and
"Python decides what is allowed to happen" cleanly separated.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

# Matches <file path="..."> ... </file> (attributes in either order, optional)
_FILE_OPEN = re.compile(
    r'<file\s+path\s*=\s*(?P<q1>["\'])(?P<path>.+?)(?P=q1)\s*>', re.IGNORECASE
)
_FILE_CLOSE = re.compile(r"</file\s*>", re.IGNORECASE)
_DELETE = re.compile(
    r'<delete\s+path\s*=\s*(?P<q>["\'])(?P<path>.+?)(?P=q)\s*/?>', re.IGNORECASE
)
_NOTE = re.compile(r"<note>(?P<body>.*?)</note>", re.IGNORECASE | re.DOTALL)

# Files an agent must never create or overwrite.
PROTECTED_NAMES = {".git", ".env", ".env.local", "id_rsa", "id_ed25519", ".ssh"}
PROTECTED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt"}
PROTECTED_DIR_PREFIXES = (".git/", ".ssh/", ".aws/", ".gnupg/")


class PatchError(Exception):
    """A proposed patch was rejected. The agent should be told and retry."""


@dataclass
class FileWrite:
    path: str
    content: str
    action: str = "created"  # created | modified


@dataclass
class ParsedPatch:
    writes: list[FileWrite] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.writes and not self.deletes

    def paths(self) -> list[str]:
        return [w.path for w in self.writes] + list(self.deletes)


# ── parsing ──────────────────────────────────────────────────────────────────


def parse_patch(text: str) -> ParsedPatch:
    """
    Extract file writes, deletions and notes from an LLM reply.

    Tolerant of prose around the envelope, strict about the envelope itself.
    """
    patch = ParsedPatch(raw=text)

    # Walk the file blocks in document order so later blocks win on conflict.
    position = 0
    while True:
        open_match = _FILE_OPEN.search(text, position)
        if not open_match:
            break
        close_match = _FILE_CLOSE.search(text, open_match.end())
        if not close_match:
            break
        path = _normalise_path(open_match.group("path"))
        body = text[open_match.end() : close_match.start()]
        patch.writes.append(
            FileWrite(path=path, content=_dedent(body), action="created")
        )
        position = close_match.end()

    for match in _DELETE.finditer(text):
        patch.deletes.append(_normalise_path(match.group("path")))

    for match in _NOTE.finditer(text):
        note = match.group("body").strip()
        if note:
            patch.notes.append(note)

    # De-duplicate deletes that are also written.
    written = {w.path for w in patch.writes}
    patch.deletes = [d for d in dict.fromkeys(patch.deletes) if d not in written]
    return patch


def _dedent(text: str) -> str:
    """Strip the single blank line and uniform indentation a model tends to add."""
    text = text.strip("\n")
    lines = text.splitlines()
    if not lines:
        return ""
    indents = [
        len(line) - len(line.lstrip()) for line in lines if line.strip().startswith((" ", "\t"))
    ]
    cut = min(indents) if indents else 0
    if cut:
        lines = [line[cut:] if line.strip() else line for line in lines]
    return "\n".join(lines).strip("\n") + "\n"


_LEADING_DOT_SLASH = re.compile(r"^(?:\./)+")


def _normalise_path(raw: str) -> str:
    """
    Canonicalise a model-supplied path to a repo-relative form.

    Only `./` prefixes are stripped. Crucially, `../` and absolute paths are
    left intact so the guard in `resolve_within` / `is_protected` can see and
    reject them -- an earlier version used `lstrip('./')`, which silently
    turned `../../etc/passwd` into `etc/passwd` and `.env` into `env`,
    defeating both the traversal check and the protected-file check.
    """
    path = raw.strip().strip("\"'").replace("\\", "/").strip()
    while True:
        stripped = _LEADING_DOT_SLASH.sub("", path)
        if stripped == path:
            break
        path = stripped
    return path.strip()


def is_absolute_path(raw: str) -> bool:
    """Whether the model asked for a filesystem-absolute path (POSIX or drive)."""
    candidate = raw.strip().strip("\"'").replace("\\", "/")
    return candidate.startswith("/") or bool(re.match(r"^[A-Za-z]:/", candidate))


# ── safety ───────────────────────────────────────────────────────────────────


def is_protected(rel_path: str) -> bool:
    """
    Whether an agent is forbidden from touching this path.

    Blocks VCS internals, credentials, absolute paths and anything that tries
    to escape the workspace.
    """
    if is_absolute_path(rel_path):
        return True
    lowered = rel_path.strip().replace("\\", "/").lower()
    if not lowered:
        return True
    if ".." in lowered.split("/"):
        return True
    parts = [p for p in lowered.split("/") if p]
    name = parts[-1] if parts else ""
    if name in PROTECTED_NAMES:
        return True
    if name.startswith(".env"):
        return True
    if any(lowered.endswith(s) for s in PROTECTED_SUFFIXES):
        return True
    if any(lowered.startswith(p) for p in PROTECTED_DIR_PREFIXES):
        return True
    if ".git" in parts:
        return True
    if ".ssh" in parts or ".aws" in parts or ".gnupg" in parts:
        return True
    return False


def resolve_within(workspace: Path, rel_path: str) -> Path:
    """
    Resolve `rel_path` against `workspace`, refusing to escape it.

    This is the single guard against an agent writing outside its own worktree.
    """
    if is_absolute_path(rel_path):
        raise PatchError(
            f"Absolute paths are not allowed: {rel_path!r}. "
            "Use a path relative to the repository root."
        )
    if is_protected(rel_path):
        raise PatchError(f"Refusing to modify protected path: {rel_path}")
    if not rel_path.strip():
        raise PatchError("Refusing to modify an empty path")

    workspace = Path(workspace).resolve()
    candidate = (workspace / rel_path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        raise PatchError(
            f"Path escapes the workspace: {rel_path} -> {candidate}"
        ) from None
    return candidate


# ── applying ─────────────────────────────────────────────────────────────────


@dataclass
class ApplyResult:
    written: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def files_changed(self) -> list[str]:
        return list(self.written) + list(self.deleted)

    @property
    def had_rejections(self) -> bool:
        return bool(self.rejected)


def apply_patch(
    patch: ParsedPatch,
    workspace: Path,
    allowed_prefixes: list[str] | None = None,
    dry_run: bool = False,
) -> ApplyResult:
    """
    Apply a parsed patch inside `workspace`.

    `allowed_prefixes` optionally scopes an agent to part of the tree (e.g. an
    agent told to only touch `src/api/`). Rejected paths are reported rather
    than silently dropped, so the agent can correct itself.
    """
    result = ApplyResult(notes=list(patch.notes))
    workspace = Path(workspace)

    for write in patch.writes:
        target = _guard(write.path, workspace, allowed_prefixes, result)
        if target is None:
            continue
        action = "modified" if target.exists() else "created"
        if not dry_run:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(write.content, encoding="utf-8")
            except OSError as exc:
                # A path the filesystem refuses -- too long, illegal character,
                # name taken on Windows -- is the agent's mistake to correct, the
                # same as an out-of-scope path. Raising here killed the agent
                # outright instead of feeding the repair loop a reason.
                result.rejected.append(
                    {"path": write.path, "reason": f"cannot write file: {exc}"}
                )
                continue
        write.action = action
        result.written.append(write.path)

    for rel in patch.deletes:
        target = _guard(rel, workspace, allowed_prefixes, result)
        if target is None:
            continue
        if not dry_run and target.exists():
            try:
                target.unlink()
            except OSError as exc:
                result.rejected.append(
                    {"path": rel, "reason": f"cannot delete file: {exc}"}
                )
                continue
        result.deleted.append(rel)

    return result


def _guard(
    rel_path: str,
    workspace: Path,
    allowed_prefixes: list[str] | None,
    result: ApplyResult,
) -> Path | None:
    """Validate one path, recording a rejection instead of raising."""
    try:
        target = resolve_within(workspace, rel_path)
    except PatchError as exc:
        result.rejected.append({"path": rel_path, "reason": str(exc)})
        return None
    if allowed_prefixes:
        normalised = rel_path.replace("\\", "/")
        if not any(normalised.startswith(p) for p in allowed_prefixes):
            result.rejected.append(
                {
                    "path": rel_path,
                    "reason": f"outside the agent's allowed scope {allowed_prefixes}",
                }
            )
            return None
    return target


# ── prompt-side helpers ──────────────────────────────────────────────────────

FILE_CONTRACT = """\
To change files, emit blocks in exactly this format and nothing else:

<file path="relative/path/from/repo/root.py">
...the complete new contents of the file...
</file>

To remove a file:

<delete path="relative/path/to/remove.py"/>

Then a short explanation:

<note>What you changed, and anything left to do.</note>

Rules:
- `path` is always relative to the repository root. Never use absolute paths.
- Always emit the COMPLETE file contents, never a diff or a fragment.
- Do not attempt to edit .git internals, .env files, or any key/credential file.
- Do not write prose outside <note> blocks; it is discarded.
"""


def summarise_prompt_contract(allowed_prefixes: list[str] | None = None) -> str:
    contract = FILE_CONTRACT
    if allowed_prefixes:
        contract += (
            f"\n- You may ONLY touch these paths: {', '.join(allowed_prefixes)}\n"
        )
    return contract


def diff_summary(workspace: Path, paths: list[str]) -> dict[str, int]:
    """Crude additions/deletions per file, for the UI without invoking git."""
    stats: dict[str, int] = {}
    for rel in paths:
        target = workspace / rel
        if not target.exists():
            stats[rel] = 0
            continue
        try:
            stats[rel] = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            stats[rel] = 0
    return stats


def shell_command(command: str) -> list[str]:
    """Split a command string safely, for prompts that suggest one."""
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def ensure_workspace(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def workspace_relative(workspace: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(workspace).resolve()))
    except ValueError:
        return os.path.basename(str(path))
