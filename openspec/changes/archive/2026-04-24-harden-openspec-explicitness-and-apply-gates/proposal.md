## Why

This repository is using OpenSpec for consequential workflow and analysis changes, but the current proposal/apply flow still allows avoidable ambiguity: names left implicit, vague placeholders, and open questions that are not clearly classified as blockers versus implementation-time audits. That creates preventable drift between what the user thinks was decided and what the implementation ends up doing.

This should be hardened before executing larger workflow changes such as the segmentation/quantification split. If the repo wants future proposals to converge on maximum explicitness by default, that behavior needs to live in repo-local OpenSpec policy, repo-local skills, and a validation gate rather than in one-off prompting.

## What Changes

- Add a repository capability that governs how OpenSpec changes in this repo must express explicit decisions, deferred decisions, and pre-implementation blockers.
- Require proposals and designs to prefer exact workflow IDs, module paths, function names, config filenames, CLI commands, and output roots whenever they can be decided at spec time.
- Require open questions to be classified explicitly as `resolve_before_apply`, `audit_first_then_decide`, or `defer_ok` rather than living in one ambiguous bucket.
- Add an apply-time gate so implementation stops when unresolved `resolve_before_apply` questions remain.
- Add a repo-local explicitness checker that can fail validation when a change still contains unresolved blocking questions or high-risk vague placeholders.
- Update repo-local OpenSpec skills so proposal creation includes an explicitness self-review and apply flow includes an open-question review before implementation.

## Explicit Decisions

- The governance will be encoded in four repo-local layers:
  - `openspec/config.yaml`
  - `AGENTS.md`
  - `.codex/skills/openspec-propose/SKILL.md`
  - `.codex/skills/openspec-apply-change/SKILL.md`
- The repo-local explicitness checker path is `scripts/check_openspec_explicitness.py`.
- The explicit open-question status tags are exactly:
  - `[resolve_before_apply]`
  - `[audit_first_then_decide]`
  - `[defer_ok]`

## Capabilities

### New Capabilities
- `openspec-change-governance`: Define repository rules for proposal explicitness, open-question classification, and apply-time implementation gates.

### Modified Capabilities

## Impact

- Affected config and policy files: `openspec/config.yaml` and `AGENTS.md`.
- Affected repo-local skills: `.codex/skills/openspec-propose/SKILL.md` and `.codex/skills/openspec-apply-change/SKILL.md`.
- Affected validation surface: a new repo-local checker such as `scripts/check_openspec_explicitness.py`, plus tests and task-level validation commands that invoke it.
- Affected workflow: future OpenSpec proposals in this repo will be expected to carry an explicit decision register and tagged open questions before they are treated as apply-ready.
