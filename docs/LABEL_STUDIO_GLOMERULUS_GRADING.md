# Label Studio Glomerulus-Instance Grading

This document describes the Stage 1 Label Studio grading contract for complete glomeruli. It is a data-collection and validation contract, not a model-assisted second-review workflow.

## One-Command Local Startup

Start from a directory of images:

```bash
eq labelstudio start --images /path/to/images
```

The command recursively imports `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff` files, starts a local Docker Label Studio instance, creates or reuses the `EQ Glomerulus Grading` project, applies `configs/label_studio_glomerulus_grading.xml`, imports the image tasks, and prints the Label Studio URL plus the project URL.

To inspect the plan without starting Docker or calling Label Studio:

```bash
eq labelstudio start --images /path/to/images --dry-run
```

Useful options:

```bash
eq labelstudio start \
  --images /path/to/images \
  --project-name "Kidney Glomerulus Grading" \
  --port 8080 \
  --runtime-root /Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/labelstudio
```

The command is local/admin bootstrap only. It does not run segmentation, quantification, MedSAM/SAM, model second review, or adjudication.

## Purpose

The lab workflow should grade complete glomeruli, not whole images. Each human grade must be linked to the exact glomerulus region it describes so `eq` can later support second review, adjudication, grader variability analysis, and animal-level rollups.

Historical image-level scores remain useful as legacy baseline data. They must not be treated as per-glomerulus ground truth because the assignment between each grade and each glomerulus was not preserved.

## Collaborator Workflow

Collaborators use Label Studio only:

1. Open the assigned image in Label Studio.
2. Mark each complete glomerulus with the `complete_glomerulus` region label.
3. Select that region and assign exactly one `endotheliosis_grade`.
4. Mark cut-off or partial glomeruli with `cutoff_partial_glomerulus`.
5. Do not grade cut-off or partial glomeruli.
6. Submit the task.

Primary grading is human-first and model-blind. The primary Label Studio config intentionally does not expose model grade, confidence, disagreement, or decision-state fields.

## Admin And Developer Boundary

Use `eq labelstudio start --images /path/to/images` as the primary setup path. Use `configs/label_studio_glomerulus_grading.xml` only when inspecting or manually debugging the Label Studio config.

By default, bootstrap runtime artifacts live under:

```text
<active-runtime-root>/labelstudio/
```

That runtime tree contains Label Studio data, generated import manifests, and bootstrap metadata. Keep those generated artifacts out of Git.

The initial ingestion authority is a Label Studio JSON export. The repo-owned parser lives in `src/eq/labelstudio/glomerulus_grading.py` and validates that:

- every complete glomerulus grade is linked to one region ID
- every complete glomerulus has exactly one grade from each grader pass
- cut-off or partial glomeruli have no grade
- named grader provenance is present from Label Studio annotation identity
- legacy image-level average scores are rejected for this contract

The parser emits glomerulus-level records and rollup-ready records in memory. Persist generated CSVs under configured runtime or output roots when adding command wrappers; do not commit raw Label Studio exports or generated grading outputs.

## Record Identity

The atomic unit is:

```text
image_id + glomerulus_instance_id
```

The stable source record ID used by the Stage 1 parser is:

```text
image_id::glomerulus_instance_id::annotation_id::grader_user_id
```

Rollups must preserve this source record ID so image, kidney, or animal averages can be traced back to included complete glomeruli.

## Not Yet Implemented

The following are intentionally deferred to later OpenSpec changes:

- model second-review inference
- second-review or adjudication Label Studio project creation
- live Label Studio ML backend deployment
- MedSAM/SAM-assisted glomerulus selection
- collaborator-visible model suggestions during primary grading
