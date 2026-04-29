from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'check_openspec_explicitness.py'


def _write_change(root: Path, *, proposal: str, design: str) -> Path:
    change_dir = root / 'change'
    change_dir.mkdir(parents=True, exist_ok=True)
    (change_dir / 'proposal.md').write_text(proposal, encoding='utf-8')
    (change_dir / 'design.md').write_text(design, encoding='utf-8')
    return change_dir


def _run_checker(change_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(change_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_checker_accepts_compliant_change(tmp_path: Path):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Example.

## Explicit Decisions

- Use `workflow: exact_name`.
""",
        design="""## Context

Example.

## Explicit Decisions

- Runner is `src/eq/example.py`.

## Open Questions

- [audit_first_then_decide] Inspect the live CLI help to decide whether the old wrapper is still semantically useful.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 0, result.stderr
    assert 'passed' in result.stdout.lower()


def test_checker_fails_on_blocking_question(tmp_path: Path):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Example.

## Explicit Decisions

- Exact name chosen.

## Open Questions

- [resolve_before_apply] Should the workflow ID be `a` or `b`?
""",
        design="""## Context

Example.

## Explicit Decisions

- Keep the CLI.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 1
    assert 'blocking_open_question' in result.stderr


def test_checker_fails_on_untagged_open_question(tmp_path: Path):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Example.

## Explicit Decisions

- Exact name chosen.

## Open Questions

- Should we keep the old wrapper?
""",
        design="""## Context

Example.

## Explicit Decisions

- Keep the runner exact.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 1
    assert 'untagged_open_question' in result.stderr


def test_checker_fails_on_vague_placeholder(tmp_path: Path):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Example.

## Explicit Decisions

- We will add a new workflow runner.
""",
        design="""## Context

Example.

## Explicit Decisions

- Use the exact CLI.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 1
    assert 'vague_placeholder' in result.stderr


def test_checker_fails_execution_surface_change_without_logging_and_docs_notes(
    tmp_path: Path,
):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Update `src/eq/run_config.py` for a new workflow.

## Explicit Decisions

- Touch `src/eq/run_config.py`.
""",
        design="""## Context

Execution change.

## Explicit Decisions

- Keep `eq run-config`.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 1
    assert 'missing_logging_contract_note' in result.stderr
    assert 'missing_docs_impact_note' in result.stderr


def test_checker_accepts_execution_surface_change_with_logging_and_docs_notes(
    tmp_path: Path,
):
    change_dir = _write_change(
        tmp_path,
        proposal="""## Why

Update `src/eq/run_config.py` for a new workflow.

## Explicit Decisions

- Touch `src/eq/run_config.py`.
- logging-contract: classify the new workflow as `entrypoint_capture` and validate its runtime log.
- docs-impact: update operator docs for the new runtime log path.
""",
        design="""## Context

Execution change.

## Explicit Decisions

- Keep `eq run-config`.
""",
    )
    result = _run_checker(change_dir)
    assert result.returncode == 0, result.stderr
