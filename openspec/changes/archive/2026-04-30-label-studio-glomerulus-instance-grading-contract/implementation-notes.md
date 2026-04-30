# Implementation Notes

## Reuse Audit

### Checked Existing Owners

- `src/eq/quantification/labelstudio_scores.py`
  - Reusable: `_load_annotation_payload`, `_normalize_image_name`, `_coerce_supported_grade`, `extract_label_studio_grade`, `LabelStudioScoreError`.
  - Not sufficient alone: this module intentionally recovers one image-level score per task/image and selects latest image-level annotations. It does not preserve per-region grade linkage, multiple glomeruli per image, cutoff exclusions, or named grader records.
- `src/eq/quantification/cohorts.py`
  - Reusable concept: `_load_label_studio_choice_scores` consumes Label Studio exports and preserves task/annotation fields for image-level cohort building.
  - Not sufficient alone: it explicitly drops to one score row per task/image and is coupled to current cohort admission flows.
- `src/eq/quantification/input_contract.py`
  - Reusable concepts: fail-closed contract errors, provenance references, stable path validation, and grouping identity checks.
  - Not sufficient alone: the glomerulus-instance contract has a different row identity, `image_id + glomerulus_instance_id`, and must validate region/grade linkage before downstream quantification.

### New Owner Justification

`src/eq/labelstudio/` is justified as a new owner because this change introduces Label Studio region-level annotation parsing and validation that is distinct from existing image-level quantification score recovery. It should still reuse existing score coercion and payload loading rather than duplicating those concepts.

### Initial Ingestion Source Decision

Use Label Studio JSON export as the first ingestion authority for Stage 1. It is file-based, reproducible in tests, and can preserve task, annotation, user, region, and per-region result IDs without requiring live Label Studio access. SDK/API sync and webhooks can be added later after the export contract is stable.

### Label Studio Region Linkage Evidence

Label Studio supports per-region controls with `perRegion="true"`. Annotation exports represent region-linked results by sharing the same result `id` between the image region and its per-region metadata or choice result. Stage 1 should validate this shared result ID rather than relying on result order.
