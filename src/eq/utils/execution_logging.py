"""Execution-scoped runtime logging helpers."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence, TextIO

from eq.utils.paths import get_logs_path

_ACTIVE_CONTEXT: contextvars.ContextVar["ExecutionLogContext | None"] = (
    contextvars.ContextVar("eq_execution_log_context", default=None)
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ExecutionLogContext:
    """Execution-scoped log metadata."""

    surface: str
    run_id: str
    log_path: Path
    runtime_root: Path | None = None
    dry_run: bool = False
    config_path: Path | None = None
    command: Sequence[str] | None = None
    run_id_source: str = "provided"
    started_at: str = ""


def current_execution_log_context() -> ExecutionLogContext | None:
    """Return the active execution logging context, if one exists."""
    return _ACTIVE_CONTEXT.get()


def execution_logging_is_active() -> bool:
    """Return whether execution-scoped logging is active."""
    return current_execution_log_context() is not None


@contextlib.contextmanager
def runtime_root_environment(runtime_root: str | Path) -> Iterator[None]:
    """Temporarily expose a runtime root to path helpers that read the environment."""
    previous = os.environ.get("EQ_RUNTIME_ROOT")
    os.environ["EQ_RUNTIME_ROOT"] = str(Path(runtime_root).expanduser())
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EQ_RUNTIME_ROOT", None)
        else:
            os.environ["EQ_RUNTIME_ROOT"] = previous


def _safe_component(value: str, *, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or "/" in text
        or "\\" in text
        or text in {".", ".."}
        or ".." in Path(text).parts
        or not _SAFE_COMPONENT.match(text)
    ):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return text


def derive_run_id(
    *,
    explicit_run_id: str | None = None,
    config_run_name: str | None = None,
    output_stem: str | Path | None = None,
    timestamp: str | None = None,
) -> tuple[str, str]:
    """Derive a stable run id and return ``(run_id, source)``."""
    candidates = (
        ("run_id", explicit_run_id),
        ("config_run_name", config_run_name),
        ("output_stem", Path(output_stem).stem if output_stem is not None else None),
    )
    for source, value in candidates:
        if value not in (None, ""):
            return _safe_component(str(value), label="run_id"), source
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return _safe_component(f"run_{stamp}", label="run_id"), "generated_timestamp"


def resolve_execution_log_path(
    *,
    surface: str,
    run_id: str,
    mode: str,
    timestamp: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Resolve a durable execution log path beneath the canonical logs root."""
    safe_run_id = _safe_component(run_id, label="run_id")
    stamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = "_dry_run" if dry_run else ""
    if mode == "run_config":
        return get_logs_path() / "run_config" / safe_run_id / f"{stamp}{suffix}.log"
    if mode == "direct":
        safe_surface = _safe_component(surface, label="surface")
        return get_logs_path() / "direct" / safe_surface / safe_run_id / f"{stamp}{suffix}.log"
    raise ValueError(f"Unsupported execution log mode: {mode!r}")


def _format_command(command: Sequence[str] | None) -> str:
    if not command:
        return ""
    return " ".join(shlex.quote(str(part)) for part in command)


def log_execution_start(
    logger: logging.Logger, context: ExecutionLogContext, *, workflow: str | None = None
) -> None:
    """Log a standard execution-start record."""
    logger.info("SURFACE=%s", context.surface)
    if workflow:
        logger.info("WORKFLOW=%s", workflow)
    logger.info("RUN_ID=%s", context.run_id)
    logger.info("RUN_ID_SOURCE=%s", context.run_id_source)
    logger.info("DRY_RUN=%s", context.dry_run)
    logger.info("LOG_PATH=%s", context.log_path)
    logger.info("PYTHON=%s", sys.executable)
    if context.runtime_root is not None:
        logger.info("RUNTIME_ROOT=%s", context.runtime_root)
    if context.config_path is not None:
        logger.info("CONFIG=%s", context.config_path)
    command = _format_command(context.command)
    if command:
        logger.info("COMMAND=%s", command)
    logger.info("EXECUTION_STATUS=started")


def log_execution_success(logger: logging.Logger, elapsed: float) -> None:
    """Log a standard execution-success record."""
    logger.info("EXECUTION_STATUS=completed")
    logger.info("ELAPSED_SECONDS=%.3f", elapsed)


def log_execution_failure(logger: logging.Logger, elapsed: float, exc: BaseException) -> None:
    """Log a standard execution-failure record."""
    logger.error("EXECUTION_STATUS=failed")
    logger.error("ELAPSED_SECONDS=%.3f", elapsed)
    logger.error("EXCEPTION_TYPE=%s", type(exc).__name__)
    logger.error("EXCEPTION=%s", exc)


@contextlib.contextmanager
def execution_log_context(
    context: ExecutionLogContext,
    *,
    logger_name: str = "eq",
    workflow: str | None = None,
    level: int = logging.INFO,
) -> Iterator[ExecutionLogContext]:
    """Attach a temporary execution file handler and clean it up afterwards."""
    logger = logging.getLogger(logger_name)
    context.log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(context.log_path)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    handler._eq_execution_handler = True  # type: ignore[attr-defined]
    token = _ACTIVE_CONTEXT.set(context)
    started = time.time()
    logger.addHandler(handler)
    if logger.level > level:
        logger.setLevel(level)
    try:
        log_execution_start(logger, context, workflow=workflow)
        yield context
    except BaseException as exc:
        log_execution_failure(logger, time.time() - started, exc)
        raise
    else:
        log_execution_success(logger, time.time() - started)
    finally:
        logger.removeHandler(handler)
        handler.close()
        _ACTIVE_CONTEXT.reset(token)


def make_execution_log_context(
    *,
    surface: str,
    mode: str,
    explicit_run_id: str | None = None,
    config_run_name: str | None = None,
    output_stem: str | Path | None = None,
    runtime_root: str | Path | None = None,
    dry_run: bool = False,
    config_path: str | Path | None = None,
    command: Sequence[str] | None = None,
) -> ExecutionLogContext:
    """Create an execution log context using repo path contracts."""
    run_id, run_id_source = derive_run_id(
        explicit_run_id=explicit_run_id,
        config_run_name=config_run_name,
        output_stem=output_stem,
    )
    log_path = resolve_execution_log_path(
        surface=surface,
        run_id=run_id,
        mode=mode,
        dry_run=dry_run,
    )
    return ExecutionLogContext(
        surface=_safe_component(surface, label="surface"),
        run_id=run_id,
        log_path=log_path,
        runtime_root=Path(runtime_root).expanduser() if runtime_root is not None else None,
        dry_run=dry_run,
        config_path=Path(config_path).expanduser() if config_path is not None else None,
        command=list(command) if command is not None else None,
        run_id_source=run_id_source,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )


@contextlib.contextmanager
def direct_execution_log_context(
    *,
    surface: str,
    explicit_run_id: str | None = None,
    config_run_name: str | None = None,
    output_stem: str | Path | None = None,
    runtime_root: str | Path | None = None,
    dry_run: bool = False,
    config_path: str | Path | None = None,
    command: Sequence[str] | None = None,
    workflow: str | None = None,
    logger_name: str = "eq",
) -> Iterator[ExecutionLogContext]:
    """Attach direct execution logging unless a parent execution context exists."""
    active = current_execution_log_context()
    if active is not None:
        yield active
        return
    context = make_execution_log_context(
        surface=surface,
        mode="direct",
        explicit_run_id=explicit_run_id,
        config_run_name=config_run_name,
        output_stem=output_stem,
        runtime_root=runtime_root,
        dry_run=dry_run,
        config_path=config_path,
        command=command,
    )
    with execution_log_context(context, logger_name=logger_name, workflow=workflow) as active_context:
        yield active_context


def run_logged_subprocess(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
    console: TextIO | None = None,
) -> None:
    """Run a subprocess while teeing stdout and stderr through logging."""
    logger = logger or logging.getLogger("eq.subprocess")
    console = console or sys.stdout
    command_text = _format_command(command)
    logger.info("SUBPROCESS_COMMAND=%s", command_text)
    print(command_text, flush=True, file=console)
    if dry_run:
        logger.info("SUBPROCESS_STATUS=dry_run")
        return
    started = time.time()
    process = subprocess.Popen(
        list(command),
        env=dict(env) if env is not None else os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip("\n")
        logger.info("SUBPROCESS_OUTPUT=%s", text)
        print(text, flush=True, file=console)
    return_code = process.wait()
    logger.info("SUBPROCESS_RETURN_CODE=%s", return_code)
    logger.info("SUBPROCESS_ELAPSED_SECONDS=%.3f", time.time() - started)
    if return_code:
        logger.error("SUBPROCESS_STATUS=failed")
        raise subprocess.CalledProcessError(return_code, list(command))
    logger.info("SUBPROCESS_STATUS=completed")
