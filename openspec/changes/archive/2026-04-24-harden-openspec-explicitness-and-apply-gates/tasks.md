## 1. Harden repo-local OpenSpec policy surfaces

- [x] 1.1 Update `openspec/config.yaml` to add explicitness rules for proposal and design artifacts, including the required `## Explicit Decisions` / `## Open Questions` convention and the allowed question status tags.
- [x] 1.2 Update `AGENTS.md` with a repository OpenSpec governance section that states maximum explicitness is the default and that unresolved `resolve_before_apply` questions block implementation.

## 2. Harden repo-local proposal and apply behavior

- [x] 2.1 Update `.codex/skills/openspec-propose/SKILL.md` so proposal generation includes an explicitness self-review that surfaces missing exact names, unresolved blocking questions, and ambiguous open-question status before the change is presented as ready.
- [x] 2.2 Update `.codex/skills/openspec-apply-change/SKILL.md` so apply flow reviews open questions before implementation and pauses automatically when unresolved `[resolve_before_apply]` items remain.
- [x] 2.3 Ensure the repo-local apply guidance still allows implementation to proceed when remaining questions are limited to `[audit_first_then_decide]` or `[defer_ok]`, while surfacing them explicitly.

## 3. Add an explicitness checker and focused tests

- [x] 3.1 Add `scripts/check_openspec_explicitness.py` to validate active changes for the repo-local explicitness contract.
- [x] 3.2 Make the checker fail on unresolved `[resolve_before_apply]` questions, untagged open questions, and high-risk ambiguity patterns in active change artifacts.
- [x] 3.3 Add focused automated tests for compliant changes, blocking unresolved questions, untagged questions, and representative vague placeholder failures.

## 4. Wire the checker into validation and examples

- [x] 4.1 Update active-change validation guidance or example tasks so repository changes can invoke the explicitness checker during final verification.
- [x] 4.2 Add or update one representative active change artifact to follow the new explicitness convention if needed to prove the checker and guidance are usable in practice.
- [x] 4.3 Run targeted validation for the new governance surface, including the explicitness checker tests and `env OPENSPEC_TELEMETRY=0 openspec validate harden-openspec-explicitness-and-apply-gates --strict`.
