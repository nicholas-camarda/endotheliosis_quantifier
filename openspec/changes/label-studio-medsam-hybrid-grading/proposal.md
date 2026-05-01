## Why

Fine-tuned MedSAM generated-mask releases from `medsam-glomeruli-fine-tuning` will be scientifically ready for labeling only if graders can reliably turn proposals into finalized glomerulus regions and attach endotheliosis scores per region inside Label Studio, with minimal CLI friction.

## What Changes

- Add a dedicated integration capability for **hybrid** grading: preload MedSAM proposal regions from a configured mask release while allowing human edit/delete/add—including **box-assisted** MedSAM for missing proposals—then export one authoritative row per scored complete-glomerulus region with full lineage (mask release, proposal kind, edit state, versioning fields).
- Simplify collaborator-facing invocation: `**eq labelstudio start <image-dir>`** as the advertised entrypoint; optional YAML config (`configs/` path TBD by design) carries MedSAM mask release binding, ML backend endpoint, project defaults, and policy flags—not a growing list of CLI flags for standard users.
- Extend local bootstrap gates: optional preload-from-release validation; startup health-check for ML box-assist when required for hybrid mode; preserve existing Docker-backed Label Studio model.
- Extend glomerulus grading export contract parsers/validators with region lineage enums, latest-vs-history provenance hooks, and fail-closed rules aligned to per-region Choices.

## Capabilities

### New Capabilities

- `label-studio-medsam-hybrid-grading`: Label Studio UX and data contracts for preload + edit + box-assisted MedSAM regions, authoritative per-region grading, ML backend coupling, versioning/provenance semantics, and config-minimized CLI surfaces.

### Modified Capabilities

- `label-studio-local-bootstrap`: Collaborator CLI shape (positional `<image-dir>`), YAML-backed parameters, optional MedSAM preload from `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/`, startup health gates for hybrid mode including ML backend.
- `label-studio-glomerulus-grading`: Exported record shape and validation rules to include mask release lineage, proposal kind (`auto` vs `box_assisted`), human edit classification, versioning fields for latest authoritative grade/mask references, consistent `region_id` / `glomerulus_instance_id`.
- `medsam-glomeruli-fine-tuning`: Generated-mask releases intended for Label Studio hybrid review MUST expose import-ready manifest fields and stable release identifiers consumable by bootstrap preload construction.

## Impact

- Affected code (expected owners): `src/eq/__main__.py`, `src/eq/labelstudio/bootstrap.py`, `src/eq/labelstudio/glomerulus_grading.py` (or successor module naming), configs under `configs/`, new or extended unit tests mirroring fixtures.
- Affected integrations: Label Studio labeling config XML (region types, preannotations wiring), ML backend companion service boundary (implement or wrap—not vendored inside `eq` unless a later architectural change says otherwise).
- Affected artifact contracts: consumes `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` and registry `derived_data/generated_masks/glomeruli/manifest.csv` as documented by `medsam-glomeruli-fine-tuning`.
- Dependency ordering: assumes `medsam-glomeruli-fine-tuning` applies first for any mask release artifact contract this change binds to.

## Explicit Decisions

- Change name (OpenSpec folder): `label-studio-medsam-hybrid-grading`.
- New capability id: `label-studio-medsam-hybrid-grading`.
- Collaborator CLI primary form: `**eq labelstudio start <image-dir>**` where `<image-dir>` is a filesystem path argument (POSIX friendly); YAML config resolves MedSAM preload and companion service settings unless explicitly overridden via **limited** `--config`/`EQ_`* for admins only.
- Cold-import policy (**B**): images may ship with zero preload masks; graders recover via box-assisted MedSAM or manual contours; bootstrap does not refuse the project solely for absent preload predictions.
- Box-assist is a **production dependency** whenever hybrid grading is enabled: health-check before usable labeling; labeling cannot silently pretend multi-glomerulus work remains easy if companion is unreachable (fail closed for that labeling session/profile).
- Versioning stance: authoritative exports expose **latest** committed annotation state with explicit lineage fields permitting audit; finer-grained LC history deferred unless LS export makes it trivial.
- Release-selection stance: implementation MUST audit the generated-mask registry first and choose the latest eligible MedSAM release by explicit ordering rules (status + timestamp), not by assuming a fixed release name.
- Required region source states for downstream ingestion: `auto_medsam`, `human_edited_medsam`, `box_assisted_medsam`, `excluded_partial_glomerulus`, and `rejected_medsam_candidate`.
- **Brush-first UX:** Final glomerulus regions remain **brush-editable** in Label Studio; box-assist supplies MedSAM prompts for new or missing coverage, not rectangle-only graded ROIs.
- **Use-case balance:** **Box-assist** is essential for **novel** images/glomeruli without preload; **preload + repeat labeling** of the same tasks remains a supported path (re-review, QA, updated grades).
- **No legacy score decomposition:** This change does **not** introduce models or heuristics that infer per-glomerulus scores from historical image-level aggregate grades (see `endotheliosis-grading-quant-learning-loop` for cohort-era flexibility).

## Open Questions

- [audit_first_then_decide] Exact Label Studio primitives for preload + box-assisted round-trip (`precomputed/predictions` vs ML backend predictor API vs custom storage link). Audit current Label Studio version in `heartexlabs/label-studio:latest` and existing repo pilot code before locking implementation.
- [defer_ok] Whether administrators ever need a second canned YAML profile beside `configs/label_studio_medsam_hybrid.yaml`; v1 ships a single authoritative example.
- [defer_ok] Whether v1 bundles the ML companion as a sibling Docker Compose service beside Label Studio versus a separate conda-invoked subprocess—operational topology only affects packaging, not conceptual contract above.

## logging-contract

This change extends `eq labelstudio start` and companion operators only. Durable execution logging remains the existing `eq` CLI log stream; no new repository log root or subprocess tee contract is introduced. The MedSAM companion MAY emit its own local logs under operator control and MUST NOT write into Git-tracked paths.

## docs-impact

Update `README.md` and `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md` with the positional `eq labelstudio start <image-dir>` story, hybrid YAML location, and companion health expectations when implementation lands. Add git-tracked `configs/label_studio_medsam_hybrid.yaml` (template or fully documented example) as part of the implementation tasks.