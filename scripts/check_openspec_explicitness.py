#!/usr/bin/env python3
"""Validate repo-local OpenSpec explicitness conventions for active changes."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


QUESTION_TAGS = {
    "resolve_before_apply",
    "audit_first_then_decide",
    "defer_ok",
}
VAGUE_PATTERNS = (
    re.compile(r"\bor its replacement\b", re.IGNORECASE),
    re.compile(r"\bwhere needed\b", re.IGNORECASE),
    re.compile(r"\bnew workflow runner\b", re.IGNORECASE),
    re.compile(r"\bnew config\b", re.IGNORECASE),
    re.compile(r"\bnew module\b", re.IGNORECASE),
)


@dataclass
class Finding:
    level: str
    kind: str
    path: Path
    lineno: int
    message: str


def _resolve_change_path(target: str, repo_root: Path) -> Path:
    raw = Path(target)
    if raw.exists():
        return raw.resolve()
    candidate = repo_root / "openspec" / "changes" / target
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find active change {target!r}. Expected a path or openspec/changes/<name>."
    )


def _iter_section_lines(lines: list[str], heading: str) -> list[tuple[int, str]]:
    target = heading.strip().lower()
    in_section = False
    selected: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lower() == target
            continue
        if in_section:
            selected.append((idx, line.rstrip("\n")))
    return selected


def _check_explicit_decisions(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    if not any(line.strip().lower() == "## explicit decisions" for line in lines):
        findings.append(
            Finding(
                level="error",
                kind="missing_explicit_decisions",
                path=path,
                lineno=1,
                message="Missing required `## Explicit Decisions` section.",
            )
        )
    return findings


def _check_open_questions(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    section_lines = _iter_section_lines(lines, "## Open Questions")
    if not section_lines:
        return findings

    saw_question = False
    for lineno, line in section_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("<!--"):
            continue
        if stripped.startswith("- "):
            saw_question = True
            match = re.match(
                r"^-\s+\[(resolve_before_apply|audit_first_then_decide|defer_ok)\]\s+.+$",
                stripped,
            )
            if not match:
                findings.append(
                    Finding(
                        level="error",
                        kind="untagged_open_question",
                        path=path,
                        lineno=lineno,
                        message="Open question must start with one explicit status tag.",
                    )
                )
                continue
            tag = match.group(1)
            if tag == "resolve_before_apply":
                findings.append(
                    Finding(
                        level="error",
                        kind="blocking_open_question",
                        path=path,
                        lineno=lineno,
                        message="Unresolved `[resolve_before_apply]` question blocks apply.",
                    )
                )
            if tag == "audit_first_then_decide":
                lowered = stripped.lower()
                if not any(token in lowered for token in ("audit", "inspect", "evidence", "review", "check")):
                    findings.append(
                        Finding(
                            level="error",
                            kind="audit_question_missing_decider",
                            path=path,
                            lineno=lineno,
                            message="`[audit_first_then_decide]` question should name the audit target or deciding evidence.",
                        )
                    )
        else:
            findings.append(
                Finding(
                    level="error",
                    kind="untagged_open_question",
                    path=path,
                    lineno=lineno,
                    message="Non-empty content in `## Open Questions` must be expressed as tagged question bullets.",
                )
            )
    if not saw_question:
        findings.append(
            Finding(
                level="error",
                kind="empty_open_questions",
                path=path,
                lineno=section_lines[0][0],
                message="`## Open Questions` exists but does not contain any tagged questions.",
            )
        )
    return findings


def _check_vague_patterns(path: Path, lines: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(lines, start=1):
        lowered = line.strip().lower()
        if lowered.startswith("```"):
            continue
        for pattern in VAGUE_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        level="error",
                        kind="vague_placeholder",
                        path=path,
                        lineno=lineno,
                        message=f"High-risk ambiguity pattern matched `{pattern.pattern}`.",
                    )
                )
    return findings


def collect_findings(change_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    for rel_path in ("proposal.md", "design.md"):
        path = change_dir / rel_path
        if not path.exists():
            findings.append(
                Finding(
                    level="error",
                    kind="missing_artifact",
                    path=path,
                    lineno=1,
                    message="Required artifact does not exist.",
                )
            )
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        findings.extend(_check_explicit_decisions(path, lines))
        findings.extend(_check_open_questions(path, lines))
        findings.extend(_check_vague_patterns(path, lines))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate repo-local OpenSpec explicitness conventions for an active change."
    )
    parser.add_argument(
        "target",
        help="Active change name under openspec/changes/ or a direct path to the change directory.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    change_dir = _resolve_change_path(args.target, repo_root)
    findings = collect_findings(change_dir)
    if not findings:
        print(f"OpenSpec explicitness check passed for {change_dir.name}")
        return 0

    for finding in findings:
        rel_path = finding.path.relative_to(repo_root) if finding.path.is_relative_to(repo_root) else finding.path
        print(
            f"{finding.level.upper()} {finding.kind} {rel_path}:{finding.lineno} {finding.message}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
