## Context

`medsam-glomeruli-fine-tuning` will emit reusable generated-mask releases under `derived_data/generated_masks/glomeruli/medsam_finetuned/<mask_release_id>/` with registry rows in `derived_data/generated_masks/glomeruli/manifest.csv`. Collaborators already use `eq labelstudio start` for Docker-backed Label Studio and per-instance grading XML. The hybrid workflow must connect those releases into Label Studio tasks as **regions** grader edits, supplemented by interactive **box-assisted** MedSAM, while keeping human-first scoring semantics and tightening export lineage so models are trained from honest segmentation provenance rather than naive auto-masks.

## Goals / Non-Goals

**Goals:**

- Consume a named MedSAM finetuned **mask_release_id** binding (via YAML—not a proliferation of collaborator CLI flags).
- Provide **preload** proposals when available without blocking labeling when preload is incomplete (cold start **B**).
- Treat box-assisted MedSAM as a required companion for hybrid usability; fail closed startup when hybrid mode mandates it vs allowing silent degradation across multi-glomerulus images.
- Extend export validation with explicit region lineage enums and latest authoritative records with reversible provenance anchors.
- Keep collaborator ergonomics anchored on `**eq labelstudio start <image-dir>`**.
- Extend rather than bypass existing owners (`src/eq/labelstudio/bootstrap.py`, `src/eq/labelstudio/glomerulus_grading.py`, `src/eq/utils/paths.py`).

**Non-Goals:**

- Performing MedSAM training or fine-tuning inside this change.
- Embedding MedSAM CUDA training code into `src/eq`; wrap external checkpoints + inference/box endpoints per upstream contracts.
- Exposing Stage 2 model grade suggestions directly into Stage 1 primary grading UX (preserve model-blind contract already spec’d).
- Committing heavyweight runtime artifacts (`derived_data/`**, checkpoints) into Git.

## Decisions

### Decision: Canonical collaborator CLI minimizes flags

**Choice:** Teach `eq labelstudio start <image-dir>` as the advertised path while retaining backward-compatible `--images` invocation already shipped for automation.

**Alternative:** Eliminate positional support and keep `--images` only → rejected due to collaborator preference.

### Decision: YAML carries integration bindings

**Choice:** Provide `configs/label_studio_medsam_hybrid.yaml` documenting `mask_release_id`, ML companion base URL, auth token reference (env-backed), Docker/LS defaults, hybrid enablement.

**Alternative:** Environment variables exclusively → simpler shell but brittle for onboarding; YAML remains authoritative with optional overrides.

### Decision: Two-phase MedSAM ingestion

**Choice:** Offline ingestion maps release manifest polygons/masks onto Label Studio `predictions`/preannotations; interactive path calls companion HTTP JSON contract with minimal payload (bbox, image locator, checkpoint ref).

**Alternative:** Only synchronous companion generation → brittle for laptops; preload gives instant UX when release exists.

### Decision: Operational posture for companion unavailability

**Choice:** Startup health ping when hybrid mode flag true; refusal to advertise ready collaborator session if ping fails unless admin explicitly enables `offline_manual_only_allowed` bypass (non-default guard).

**Alternative:** Soft degrade silently → contradicts collaborator clarity for multi-glomerulus labeling.

### Decision: Versioned audit fields without full duplication

**Choice:** Persist `annotation_id`, LS `completed_at`, `mask_release_id`, `proposal_kind`, `region_edit_state`, optional superseded references to prior annotation drafts when LS exposes them cheaply—otherwise deterministic latest-completed semantics only.

### Decision: Parsing ownership

Extend `glomerulus_grading.py` parsers (or refactor into shared labelstudio contract helpers) versus duplicating ingestion script—reuse existing ingestion surface.

## Risks / Trade-offs

- **[Risk]** Label Studio version drift changes prediction API → mitigation: pin container tag or document audited API + contract tests referencing fixture exports.
- **[Risk]** Over-specifying ML HTTP schema before companion exists → mitigation: stabilize minimal request/response in spec with explicit extension points flagged `[defer_ok]` for batching/gpu fields.
- **[Risk]** Preload polygons misaligned vs displayed image → mitigation: deterministic coordinate normalization tests + QA overlay snapshot hook.
- **[Risk]** Admin bypass flags inadvertently used in production datasets → mitigation: fail-closed default; bypass requires YAML opt-in loudly logged.

## Migration Plan

1. Land YAML config + parsers + bootstrap wiring after MedSAM releases exist (`depends_on` sequencing with fine-tuning change).
2. Introduce staging profile for collaborators; document rotation between releases referencing registry manifest.
3. Rollback by pointing YAML `mask_release_id` null or disabling hybrid block to revert manual-only Stage 1.

## Explicit Decisions

- Collaborator advertised command: `**eq labelstudio start <image-dir>`**.
- Default integration settings file path: `**configs/label_studio_medsam_hybrid.yaml`**.
- Cold start policy (**B**) remains: zero preload masks must NOT block importing tasks.
- Box-assist default posture: hybrid mode refuses open without healthy companion unless `offline_manual_only_allowed: true`.

## Open Questions

- [audit_first_then_decide] Which Label Studio data structure (`predictions`, `annotations`, imported storage) best carries vector masks for multi-region preload for our pinned image version — audit exported JSON from pilot container tasks.
- [defer_ok] Exact HTTP companion contract paths (`/healthz`, `/v1/box_infer`, etc.) — finalize when scaffolding the companion service.
- [defer_ok] Compose packaging versus standalone script to launch the companion beside the Label Studio container — operational convenience only.