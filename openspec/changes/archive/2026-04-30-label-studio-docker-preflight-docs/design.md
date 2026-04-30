## Context

The one-command Label Studio bootstrap now exists, but Docker remains the user-facing failure point. On macOS, Docker Desktop may be installed while the daemon is stopped; the command should handle that common state by opening Docker Desktop and waiting instead of telling the user to debug Docker manually.

## Goals / Non-Goals

**Goals:**

- Make `eq labelstudio start --images <dir>` auto-start installed Docker Desktop on macOS.
- Provide explicit install instructions when Docker Desktop is missing.
- Preserve dry-run as side-effect free.
- Update README with copy-paste commands.

**Non-Goals:**

- Do not silently install Docker Desktop.
- Do not require Docker for `--dry-run`.
- Do not add production/shared Label Studio hosting.

## Explicit Decisions

- Docker Desktop auto-start is macOS-specific and uses `open -a Docker`.
- The preflight waits on `docker info`, not just `docker --version`, because the CLI can exist while the daemon is stopped.
- Missing Docker Desktop on macOS reports `brew install --cask docker`.
- Non-macOS missing Docker reports a generic install/start Docker message.

## Decisions

### Start installed Docker Desktop

If `docker info` fails and `/Applications/Docker.app` exists, the command should run `open -a Docker` and poll `docker info` until the existing timeout expires.

Alternative considered: fail immediately and tell the user to start Docker. That preserves implementation simplicity but violates the one-command goal.

### Do not auto-install Docker

Installing Docker Desktop is a heavyweight system-level action that can involve licensing, security prompts, and admin expectations. The command should provide exact install instructions but not perform installation automatically.

Alternative considered: run `brew install --cask docker`. That is too intrusive for a labeling workflow command.

## Risks / Trade-offs

- Docker Desktop startup can take longer than the Label Studio timeout → Mitigation: reuse `--timeout-seconds` and provide a clear timeout error.
- `open -a Docker` may require user GUI interaction → Mitigation: report that Docker Desktop was requested and continue polling.
- Linux/WSL Docker startup differs from macOS → Mitigation: only auto-start Docker Desktop on macOS and keep other platforms explicit.
