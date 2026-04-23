## Context

The repository now enforces the correct glomeruli training-data contract: supported training uses curated paired full images under `raw_data/.../training_pairs`, dynamic patching is the only supported training mode, and promotion-gate helpers already exist for deterministic manifests, trivial baseline comparisons, and degenerate prediction checks. What is still missing is the actual candidate-comparison workflow that answers whether transfer learning from mitochondria helps, whether scratch training performs better, and whether any resulting glomeruli artifact deserves promotion.

The current evidence argues against assuming transfer is automatically beneficial. The latest compatibility-era glomeruli artifact looked degenerate under the old static-patch contract, but that failure does not prove anything about transfer versus scratch under the corrected full-image dynamic contract. The next change therefore needs to compare candidate families empirically under a single deterministic evaluation contract and end with a real promotion decision rather than another runtime-only artifact.

Implementation review after the first pass also established that the change was not actually validation-complete:
- the scratch glomeruli path was silently collapsing `crop_size` to `image_size`, so the intended large-context crop workflow was not really being exercised
- the candidate-comparison command still pointed users toward repo-local `output/...` examples even though the active working outputs on this machine belong under `~/ProjectsRuntime/endotheliosis_quantifier/output/...`
- the first `eq-mac` comparison run generated a correct failure report, but both candidate families were unavailable, so it was diagnostic evidence rather than proof that the supported comparison workflow had executed successfully end to end

## Goals / Non-Goals

**Goals:**
- Define a concrete workflow that trains and evaluates at least one transfer candidate and one scratch candidate under the current supported glomeruli training contract.
- Make the dedicated training module CLI, not an old YAML file, the canonical comparison interface for transfer and scratch candidates.
- Persist a deterministic validation manifest that both candidates use for comparison and promotion review.
- Produce a promotion report artifact that compares each candidate to trivial baselines and the current compatibility artifact when available.
- End with a decision state that is explicit and auditable: promote a candidate, promote neither, or record insufficient evidence, including an explicit tie path for scientifically indistinguishable candidates.
- Keep runtime compatibility, supported runtime status, and scientific promotion as separate artifact states.
- Remove or rewrite stale module/config/doc surfaces that still imply patch-era glomeruli workflows or config-first promotion control.

**Non-Goals:**
- Expanding the training cohort with new cloud-only scored images that do not yet have masks.
- Solving the downstream ordinal quantification stability problem in this change.
- Declaring transfer learning superior or inferior before the comparison is run.
- Changing the raw-data versus derived-data contract established in the previous change.

## Decisions

1. **Compare transfer and scratch directly under one evaluation contract.**
   - Rationale: there is no current evidence that transfer learning improves glomeruli segmentation under the corrected data contract, and no current evidence that scratch is better either. The change should answer that question empirically rather than encoding a prior belief.
   - Alternatives considered:
     - Transfer-only retraining. Rejected because it assumes benefit without evidence.
     - Scratch-only retraining. Rejected for the same reason.

2. **Use one deterministic validation manifest for promotion, even if training remains stochastic.**
   - Rationale: training can keep dynamic variation, but promotion must compare candidates on the same fixed positive, boundary, and background evidence set.
   - Additional contract:
     - category assignment must be mutually exclusive per crop so one crop cannot be counted as both `boundary` and `positive`
     - selection must prefer spreading evidence across distinct source images when sufficient coverage exists rather than filling the manifest from the first sorted image alone
     - the manifest audit written into the report must expose how many unique images and subjects are represented so narrow evidence sets are visible instead of implicit
   - Alternatives considered:
     - Reuse random validation crops from each training run. Rejected because run-to-run stochasticity would blur candidate comparison.

3. **Treat promotion as a decision artifact, not an incidental side effect of training.**
   - Rationale: a candidate should not become the promoted model merely because it was trained most recently or because it loads successfully.
   - Alternatives considered:
     - Promote the best runtime-compatible candidate automatically by Dice. Rejected because promotion also depends on baseline comparisons and non-degenerate prediction review.

4. **Allow “promote neither” as a first-class outcome.**
   - Rationale: the point of the change is to answer the model-quality question honestly. If both candidates remain weak or degenerate, the correct result is to block promotion.
   - Alternatives considered:
     - Require the change to produce a promoted model no matter what. Rejected because it would incentivize overstating evidence.

5. **Keep candidate comparison separate from data-expansion work.**
   - Rationale: cloud-only scored images without masks may matter later, but they do not directly strengthen segmentation supervision under the current contract.
   - Alternatives considered:
     - Fold new scored-only images into this promotion change. Rejected because they are not segmentation ground truth and would blur the scope.

6. **Use the dedicated training CLI as the canonical comparison interface.**
   - Rationale: the live repo executes training through `python -m eq.training.train_glomeruli`; the YAML files are only partial optional overlays and already drift from the actual transfer implementation. Promotion comparison needs one authoritative control surface.
   - Alternatives considered:
     - Treat `configs/glomeruli_finetuning_config.yaml` as the canonical contract. Rejected because the file does not fully describe the live transfer schedule and still invites config-first ambiguity.
     - Preserve multiple equally authoritative entrypoints. Rejected because it makes candidate comparison and provenance harder to audit.

7. **Pin one canonical transfer candidate and one canonical scratch candidate before comparing families.**
   - Rationale: the comparison should answer transfer-versus-scratch under one fixed recipe, not drift across ad hoc backbone or scheduler choices.
   - Canonical transfer candidate:
     - `python -m eq.training.train_glomeruli`
     - supported `raw_data/.../training_pairs` input root
     - current `resnet34` DynamicUnet code path
     - mitochondria base artifact supplied explicitly or auto-discovered only during inventory, not silently swapped during the comparison run itself
     - current encoder-only weight loading, decoder reinitialization, frozen-head stage, and unfrozen fine-tuning stage
   - Canonical scratch candidate:
     - the same training module and data contract
     - current `resnet34` DynamicUnet code path
     - explicit `--from-scratch` execution with no hidden transfer fallback
   - Alternatives considered:
     - Leave hyperparameters implied by old configs or convenience defaults. Rejected because the comparison contract would remain underspecified.
     - Introduce a new backbone such as `resnet50` for scratch only. Rejected because it would conflate candidate family with architecture change.

8. **Use a composite promotion gate instead of a single metric winner rule.**
   - Rationale: glomeruli promotion is a scientific quality decision, not a leaderboard exercise.
   - Composite gate:
     - the candidate must exceed both trivial baselines on shared-manifest Dice and Jaccard
     - the candidate must not trip degeneracy checks such as all-background or all-foreground collapse
     - the candidate must pass prediction review on deterministic positive, boundary, and background examples
     - among candidates that clear the hard gates, the report compares shared-manifest Dice and Jaccard jointly rather than promoting solely on one metric
   - Tie / insufficient-evidence rule:
     - if both candidates clear the gates and their shared-manifest Dice and Jaccard differences both remain below an explicit practical margin, the report records `insufficient_evidence` with an explicit tie reason rather than forcing a winner
     - the initial practical tie margin for the canonical implementation is an absolute difference of `0.02` or less on both shared-manifest Dice and shared-manifest Jaccard
   - Alternatives considered:
     - Promote the highest Dice candidate automatically. Rejected because it ignores baseline failure, degeneracy, and scientific usability.

9. **Make the report legible enough to support actual visual review.**
   - Rationale: a promotion report that technically embeds review panels but does not clearly label panel order, crop provenance, or per-panel metrics is not an auditable scientific artifact.
   - Consequence:
     - the HTML report must state panel order explicitly
     - each review panel must be labeled with category, subject/image identity, crop box, and per-panel metrics
     - the report must summarize manifest coverage near the top so a narrow or repetitive evidence set is visible immediately
   - Alternatives considered:
     - Keep the minimal HTML dump and rely on CSV inspection. Rejected because the report itself is supposed to support human review.

9. **Treat ties as valid research outcomes without creating two promoted defaults.**
   - Rationale: this repository is for research, so scientifically indistinguishable candidates should remain usable for downstream experiments without pretending that one is the unique promoted default.
   - Consequence:
     - a tie records `insufficient_evidence` at the promotion-decision level
     - neither candidate becomes the sole scientifically promoted default
     - both tied candidates remain explicit runtime-compatible research candidates in provenance and reporting
     - the promotion report must record the artifact paths and show that either can be passed explicitly into downstream glomeruli segmentation and endotheliosis quantification workflows
   - Alternatives considered:
     - Force a winner anyway. Rejected because it overstates evidence.
     - Mark both as promoted defaults. Rejected because it collapses the distinction between canonical promotion and research comparators.

10. **Require stale, conflicting surfaces to be cleaned up as part of the change.**
   - Rationale: old config-first and patch-era modules create avoidable ambiguity about how glomeruli training and promotion are actually controlled.
   - Cleanup target:
     - update or retire glomeruli YAML/config/module/doc surfaces that still imply patch-directory training, config-first prediction entrypoints, or automatic transfer fallback during promotion comparison
   - Alternatives considered:
     - Leave stale surfaces in place and only document the new path. Rejected because the repo would still present multiple contradictory workflows.

11. **Start with one canonical seed per family, but preserve a clean upgrade path to repeated-seed comparison.**
   - Rationale: the first implementation should stay bounded enough to ship and inspect, while the comparison/report shape should not need redesign if repeated seeds become necessary later.
   - Consequence:
     - the canonical initial comparison runs one explicit seed for transfer and one explicit seed for scratch
     - the promotion report records the seed used for each candidate
     - report schema and provenance should be structured so future repeated-seed comparisons can append per-seed candidate rows without breaking the first implementation's artifact contract
   - Alternatives considered:
     - Require three seeds per family in the first implementation. Rejected because it adds substantial runtime and coordination cost before the single-seed workflow is proven end-to-end.
     - Ignore seed provenance in the first implementation. Rejected because it would make later repeated-seed expansion harder to audit.

12. **Use the active runtime output root as the canonical default home for comparison artifacts on this machine.**
   - Rationale: the compare workflow produces research artifacts, model candidates, and review panels that belong in the active runtime tree, not in disposable `/tmp` directories or repo-local `output/` examples that currently resolve elsewhere.
   - Consequence:
     - an explicit caller-supplied `--output-dir` remains allowed
     - when no output directory is supplied, the workflow defaults to the active runtime output root's `glomeruli_candidate_comparison/` subtree
     - docs and examples for this change must stop implying that repo-local `output/...` is the canonical location on this machine
   - Alternatives considered:
     - Keep `--output-dir` required and leave location choice entirely to the user. Rejected because the current docs and local path topology make the wrong location easy to choose.
     - Default to `/tmp` for convenience. Rejected because those artifacts are not disposable smoke-test output in the real workflow.

13. **Do not treat failure-report generation as completion evidence for the change.**
   - Rationale: a comparison report where transfer and scratch both fail is useful diagnostics, but it does not prove that the intended candidate-comparison workflow actually ran successfully under the supported environment.
   - Consequence:
     - the change remains open until an unsandboxed `eq-mac` validation run executes both candidate families successfully under the active runtime contract
     - report-only runs where candidate families are unavailable are recorded as blockers, not completion evidence
   - Alternatives considered:
     - Accept any generated promotion report as sufficient validation. Rejected because the report format alone cannot distinguish successful comparison from structured failure capture.

14. **Use the supported inference preprocessing and threshold semantics for candidate comparison.**
   - Rationale: candidate comparison is supposed to evaluate the usable segmentation behavior of the trained artifacts, not a bespoke evaluation-only preprocessing path. Reusing a generic `0.5` threshold or mismatched crop preprocessing can make viable but underconfident models look fully collapsed.
   - Consequence:
     - shared-manifest candidate evaluation must use learner-consistent preprocessing
     - binary promotion metrics must use the same underconfident-model threshold semantics as the supported segmentation inference path unless a future change explicitly revises that threshold contract
   - Alternatives considered:
     - Leave comparison on a standalone `0.5` threshold with custom preprocessing. Rejected because it can invalidate the promotion report independently of the model artifacts themselves.

## Risks / Trade-offs

- [Risk] Running both transfer and scratch candidates increases compute time and may make the change heavier than a single retrain. → Mitigation: keep training bounded but use the same deterministic evaluation contract for both so the comparison remains meaningful.
- [Risk] A single-seed comparison may over-read stochastic variation. → Mitigation: make one seed per family the bounded initial requirement, record the seed explicitly in provenance, and treat small-magnitude differences inside the practical tie margin as `insufficient_evidence`.
- [Risk] The deterministic manifest could overfit candidate comparison to a small curated slice of the data. → Mitigation: require mutually exclusive background, boundary, and positive coverage, prefer image-diverse manifest selection when available, and expose unique-image / unique-subject coverage directly in the report.
- [Risk] Transfer and scratch may differ only marginally, leaving no clear winner. → Mitigation: allow “insufficient evidence” or “promote neither” as valid outcomes.
- [Risk] The repo already contains config-driven and patch-era glomeruli surfaces that can confuse users about the real promotion interface. → Mitigation: make the training module CLI canonical and clean up or rewrite conflicting stale surfaces in the same change.
- [Risk] The compatibility artifact may be unavailable or not directly comparable in some environments. → Mitigation: make it a comparison input when available, but not the sole promotion criterion.
- [Risk] A candidate may look numerically better while still producing scientifically weak segmentations. → Mitigation: preserve the current rule that prediction degeneracy and baseline failure block promotion even if aggregate metrics improve.
- [Risk] The workflow may appear validated because it writes a clean failure report even when neither candidate family executed successfully. → Mitigation: require a real unsandboxed `eq-mac` validation run with successful transfer and scratch execution before calling the change complete.
- [Risk] Repo-local path examples may send compare artifacts to the wrong storage root on this machine. → Mitigation: default compare outputs to the active runtime output root and remove repo-local `output/...` examples from touched docs.
- [Risk] Scratch training may silently use smaller-than-requested context if `crop_size` is not propagated into the dataloader and batch-size logic. → Mitigation: thread the requested crop size through scratch training, record it in provenance, and add regression coverage.
