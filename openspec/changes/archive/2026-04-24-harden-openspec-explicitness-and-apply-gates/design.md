## Context

This repository already has a strong culture of explicit workflow contracts, but today that explicitness depends too much on the quality of the current prompting and too little on durable repo-local controls. `openspec/config.yaml` provides artifact-writing context and rules, `AGENTS.md` provides repository policy, and the repo-local skills `.codex/skills/openspec-propose/SKILL.md` and `.codex/skills/openspec-apply-change/SKILL.md` shape proposal and implementation behavior. None of those surfaces currently forces a proposal to resolve avoidable ambiguity or forces implementation to stop when unresolved blocking questions remain.

The result is a predictable failure mode: proposals can still use replacement-style wording, unnamed implementation-surface placeholders, or other vague language even when an exact name could have been chosen, and "Open Questions" can still act as a mixed bucket for true blockers, audit-time decisions, and harmless deferred ideas. That is especially risky in this repo because OpenSpec changes are often used to define workflow contracts, runtime-output boundaries, and scientific-use surfaces where ambiguous naming can turn directly into user-visible drift.

This change is cross-cutting because it touches:

- repo-level OpenSpec authoring rules in `openspec/config.yaml`
- repo policy in `AGENTS.md`
- repo-local proposal/apply skills in `.codex/skills/`
- a new validation tool and tests that make the policy enforceable

## Goals / Non-Goals

**Goals:**

- Make maximum explicitness the default style for future OpenSpec changes in this repo.
- Require proposal and design artifacts to prefer exact names, paths, workflow IDs, config filenames, CLI commands, and output roots when they can be decided at spec time.
- Separate "questions that must be resolved before apply" from "questions that are acceptable to decide during implementation after an audit."
- Make repo-local apply flow stop before implementation when unresolved blocking questions remain.
- Add a lightweight explicitness checker that can be run locally and from validation tasks.
- Keep the policy repo-local so it reflects this repository's standards without pretending OpenSpec globally behaves the same way everywhere.

**Non-Goals:**

- Modifying upstream OpenSpec CLI behavior outside this repository.
- Requiring every proposal to have zero implementation-time questions; some audit-first decisions are legitimate.
- Turning explicitness checking into a full prose-quality or style linter.
- Rewriting archived changes to comply retroactively with the new convention.

## Explicit Decisions

1. **Encode the policy in four layers: config, repo policy, skills, and validation.**
   - Rationale: `openspec/config.yaml` can influence artifact quality, `AGENTS.md` can define repo expectations, repo-local skills can shape behavior at propose/apply time, and a checker script can provide actual enforcement. No single layer is sufficient by itself.
   - Alternatives considered:
   - Put everything only in `openspec/config.yaml`. Rejected because config rules influence writing but do not gate implementation.
   - Put everything only in the skills. Rejected because skills can drift from the repo policy and are easier to bypass accidentally.

2. **Use a parseable explicitness convention for open questions.**
   - Rationale: the checker needs a simple format it can detect reliably. Proposal and design artifacts should use:
     - `## Explicit Decisions`
     - `## Open Questions`
     and every open-question line must start with one of:
     - `[resolve_before_apply]`
     - `[audit_first_then_decide]`
     - `[defer_ok]`
   - Alternatives considered:
   - Free-form prose in `Open Questions`. Rejected because it is not machine-checkable.
   - A YAML frontmatter block inside artifacts. Rejected because it is heavier than needed and would be awkward to maintain in normal spec writing.

3. **Treat unresolved `resolve_before_apply` questions as an apply blocker.**
   - Rationale: if a question is important enough to block implementation, the repo-local apply flow should stop and surface it before tasks begin. That keeps the user and implementation aligned.
   - Alternatives considered:
   - Let apply continue and rely on the implementer to notice blockers manually. Rejected because it is exactly the failure mode this change is trying to remove.

4. **Allow `audit_first_then_decide` questions, but require explicit decision criteria.**
   - Rationale: some questions are legitimately impossible to answer until code or runtime surfaces are inspected. Those questions are allowed, but they must describe what evidence will decide them, so they do not become a vague escape hatch.
   - Alternatives considered:
   - Forbid all implementation-time questions. Rejected because it would force premature decisions where an audit is the correct first step.

5. **Add a repo-local checker that flags high-risk vague placeholders and unresolved blocking questions.**
   - Rationale: repo-local validation should catch at least the most consequential ambiguity classes:
     - unresolved `resolve_before_apply`
     - untagged open questions
     - vague placeholders such as replacement-style wording, location-unspecified language, unnamed module/config placeholders, or equivalent high-risk ambiguity inside active changes
   - Alternatives considered:
   - Depend only on human review. Rejected because the user explicitly wants this to become a durable default.

## Risks / Trade-offs

- [Risk] The checker could become too rigid and punish harmless drafting language. → Mitigation: target only high-risk ambiguity patterns and open-question classification, not broad prose style.
- [Risk] Repo-local skills may still be bypassed if someone writes specs manually. → Mitigation: make the checker and validation task the real gate, not the skill text alone.
- [Risk] `audit_first_then_decide` could become a loophole for unresolved design work. → Mitigation: require each such question to name the deciding audit target or evidence source explicitly.
- [Risk] Existing in-progress changes may not match the new convention immediately. → Mitigation: apply the governance going forward and update active changes opportunistically when touched, rather than trying to bulk-rewrite archives.

## Migration Plan

1. Extend `openspec/config.yaml` with repo-local explicitness and open-question classification rules.
2. Add an OpenSpec governance section to `AGENTS.md` so the repo policy is visible outside the config surface.
3. Update `.codex/skills/openspec-propose/SKILL.md` so proposal creation includes an explicitness self-review before the change is presented as ready.
4. Update `.codex/skills/openspec-apply-change/SKILL.md` so apply flow checks for unresolved `resolve_before_apply` questions before implementation starts.
5. Add `scripts/check_openspec_explicitness.py` and focused tests.
6. Update validation tasks and examples so active changes can call the checker during final verification.

## Open Questions

- [audit_first_then_decide] Should the checker scan only `proposal.md` and `design.md`, or also `tasks.md` and delta spec files for high-risk vague placeholders? The deciding criterion is whether early implementation runs show ambiguity leaking mainly from planning artifacts or also from task/spec wording.
