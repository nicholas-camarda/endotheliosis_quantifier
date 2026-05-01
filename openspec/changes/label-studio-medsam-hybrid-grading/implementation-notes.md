# label-studio-medsam-hybrid-grading implementation notes

## Task 1.1 audit: Label Studio version and export contract

Date: 2026-05-01  
Host: macOS (`eq-mac` development context)

### Audited image/version

- Local image tag previously used by bootstrap: `heartexlabs/label-studio:latest`
- Image metadata inspection:
  - `org.opencontainers.image.version = 1.23.0`
  - `org.opencontainers.image.revision = 2a9bfbcbf0a844b999de97e601d16050a893f5fb`
- Pin decision for this change:
  - `DEFAULT_DOCKER_IMAGE = heartexlabs/label-studio:1.23.0`
  - CLI default `--docker-image` updated to `heartexlabs/label-studio:1.23.0`

### API behavior audited (prediction vs annotation)

Using a temporary local container (`localhost:8099`) with token auth:

1. Create project with `Image + BrushLabels + Choices` label config.
2. Import task payload containing `data` plus `predictions`:
   - prediction `result` item uses:
     - `type = brushlabels`
     - `from_name = glomerulus_roi`
     - `to_name = image`
     - `value.format = rle`
     - `value.rle = [...]`
     - `value.brushlabels = [...]`
3. Verified import response includes `prediction_count = 1`.
4. Verified task retrieval includes top-level `predictions` list.
5. Posted a matching annotation for the imported prediction and exported project JSON.
6. Verified export task includes both:
   - `predictions` (preloaded model proposals)
   - `annotations` (human/submitted records)
7. Verified exported annotation rows include lineage hooks:
   - `parent_prediction`
   - `parent_annotation`
   - `result` entries with `brushlabels` geometry and `choices` grade link by shared `id`.

### Contract decision from audit

- Hybrid preload should be encoded in task `predictions` (not fake pre-annotations).
- Grader output remains authoritative in `annotations`.
- Export parsers must treat `predictions` and `annotations` as separate sources and rely on `annotations` as authoritative for scored rows while preserving prediction lineage where present.

### Follow-on implications for tasks 3.x/4.x

- Task builder should generate prediction payloads with Label Studio-compatible `brushlabels` + RLE fields.
- Export validation should require coherent lineage when a row claims preload/box-assist provenance.
- Parser logic should preserve `parent_prediction`, annotation ids, and region ids for training/audit traceability.

## Task 1.2 audit: generated-mask registry and latest-valid selection

Source audited: `derived_data/generated_masks/glomeruli/manifest.csv`

Observed contract fields:

- `mask_release_id`
- `generated_mask_path`
- `generation_status`
- `mask_source`
- `release_manifest_path`
- `provenance_path`

Current local rows are dominated by release `deploy_conservative_mps_glomeruli` with `generation_status=generated` and `mask_source=medsam_finetuned_glomeruli`.

Selection criteria implemented for `selection_mode: latest_valid`:

1. Group central manifest rows by `mask_release_id`.
2. Within each release, count rows where:
   - `generation_status == generated`, and
   - `generated_mask_path` exists on disk.
3. Compute release recency by max filesystem mtime of existing generated masks.
4. Select release by descending tuple:
   - `generated_count`
   - `latest_mtime`
   - `mask_release_id` (tie-break)
5. If YAML pins `mask_release_id`, fail closed when absent from registry.

Rationale:

- Works with current manifest schema without assuming an extra `created_at` column.
- Fails closed for invalid pinned releases.
- Still allows cold-start import when no release is selected or preload does not cover all images.

## Product-momentum checkpoint (mid-implementation)

Question: *Did this slice move the product measurably closer to final operator value?*  
Answer: **Yes**.

Evidence:

- Hybrid bootstrap now supports positional CLI image-dir, YAML-first config, release selection, preload prediction payloads, and companion health gating.
- Parser now emits hybrid lineage fields and rejects contradictory lineage payloads.
- Quant contract path now accepts per-glomerulus instance exports and writes lineage summaries.

## Real-data E2E evidence (MPS host)

Runtime setup used:

- Demo image subset copied from runtime manifest rows into `/tmp/eq_hybrid_demo/images/animal_1/`.
- Companion checkpoint:
  - `/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/deploy_conservative_mps_glomeruli/finetuned_evaluation/medsam_glomeruli_best_sam_state_dict.pth`

Commands run and outcomes:

1. Start companion on MPS:
   - `PYTORCH_ENABLE_MPS_FALLBACK=1 ... python -m eq.labelstudio.medsam_companion --checkpoint ... --device mps --port 8098`
   - Health check passed (`GET /healthz` -> `200`).
2. Real box assist inference:
   - `POST /v1/box_infer` on copied runtime image returned non-empty mask (`foreground_pixels=34956`, non-empty RLE).
3. Bootstrap Label Studio hybrid project:
   - `python -m eq labelstudio start /tmp/eq_hybrid_demo/images`
   - Imported 2 tasks with 2 preload predictions (runtime-default root path).
4. Create multi-ROI annotation + export:
   - Used Label Studio API to submit one preloaded ROI + one box-assisted ROI, each with a grade.
   - Export written to `/tmp/eq_hybrid_demo/project2_export.json`.
5. Quant contract ingestion with per-glom scoring:
   - `run_contract_first_quantification(..., annotation_source=/tmp/eq_hybrid_demo/project2_export.json, stop_after='contract')`
   - Outputs written under `/tmp/eq_hybrid_demo/quant_run/`.
   - `scored_examples.csv` has 2 glomerulus-instance rows from one image.
   - `lineage_summary.json` reports `scoring_unit=glomerulus_instance`, `rows=2`, `unique_instances=2`.

Issue discovered + fixed during E2E:

- Label Studio readiness polling failed on transient `ConnectionResetError`.
- Fix: `LabelStudioApiClient.wait_until_ready` now retries `ConnectionResetError` alongside existing transient HTTP startup failures.

## Resume-here note (intentionally explicit, currently half-finished)

The end-to-end dev loop is operational (MPS companion, preload import, editable annotations, per-region grading linkage, and quant ingestion), but this change is still half-finished for operator usability and preload fidelity.

Current known state:

- Preload regions are materialized into editable annotations for immediate brush/eraser editing.
- Region grading is constrained per selected region (`perRegion` + single-choice radio).
- Tiny preload connected components are filtered at import (`MIN_PRELOAD_COMPONENT_AREA_PX = 1000`).
- Quantification drops zero-area/invalid regions from Label Studio exports to avoid ghost-row contamination.
- The underlying release masks are still semantic/union artifacts, so per-instance preload quality can remain noisy for some images.

What to do next when resuming:

1. Add explicit preloading policy options in `configs/label_studio_medsam_hybrid.yaml` for:
   - minimum component area threshold
   - optional border-touch rejection
   - optional max-area sanity gate
2. Add project bootstrap option to preserve or strip prediction overlays after materialization (current manual cleanup is API-driven).
3. Add one operator-facing cleanup command that:
   - creates a fresh demo project
   - strips prediction overlays when requested
   - prints a single canonical URL to continue annotation
4. Validate on real runtime rows that each loaded region corresponds to a single meaningful glomerulus before collaborator handoff.
