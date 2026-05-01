# Label Studio MedSAM Hybrid Grading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the latest valid fine-tuned MedSAM release into Label Studio hybrid grading so collaborators can start from preloaded **brush** regions (repeat review allowed), use **box-assisted** region creation for cold coverage, and export validated per-region lineage for **per-glomerulus** scoring—without inferring per-glom grades from legacy image-level aggregates (see learning-loop specs).

**Architecture:** Extend existing `eq labelstudio start` bootstrap and `eq.labelstudio` parsing surfaces instead of introducing parallel pipelines. Keep collaborator UX on one command (`eq labelstudio start <image-dir>`), move advanced behavior to YAML config, and enforce fail-closed runtime checks for required hybrid dependencies. Preserve human-first grading while enriching export lineage fields tied to release provenance.

**Tech Stack:** Python (`argparse`, stdlib HTTP/json), Label Studio Docker API, YAML config parsing, pytest, ruff, OpenSpec.

---

### Task 1: Hybrid Config + Release Selection Core

**Files:**
- Create: `configs/label_studio_medsam_hybrid.yaml`
- Modify: `src/eq/labelstudio/bootstrap.py`
- Test: `tests/unit/test_labelstudio_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_select_latest_valid_release_when_not_pinned():
    registry = [
        {"mask_release_id": "r1", "status": "blocked", "created_at": "2026-04-30T10:00:00Z"},
        {"mask_release_id": "r2", "status": "valid", "created_at": "2026-04-30T11:00:00Z"},
        {"mask_release_id": "r3", "status": "valid", "created_at": "2026-04-30T12:00:00Z"},
    ]
    assert _select_mask_release_id(registry, pinned_id=None) == "r3"


def test_fail_when_pinned_release_missing():
    registry = [{"mask_release_id": "r2", "status": "valid", "created_at": "2026-04-30T11:00:00Z"}]
    with pytest.raises(BootstrapError, match="Pinned mask_release_id .* not found"):
        _select_mask_release_id(registry, pinned_id="missing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_select_latest_valid_release_when_not_pinned tests/unit/test_labelstudio_bootstrap.py::test_fail_when_pinned_release_missing -q`

Expected: FAIL with missing `_select_mask_release_id` helper and/or behavior mismatch.

- [ ] **Step 3: Write minimal implementation**

```python
def _select_mask_release_id(registry_rows: list[dict[str, Any]], pinned_id: str | None) -> str:
    if pinned_id:
        if any(row.get("mask_release_id") == pinned_id for row in registry_rows):
            return pinned_id
        raise BootstrapError(f"Pinned mask_release_id {pinned_id!r} not found in registry")

    valid = [r for r in registry_rows if str(r.get("status", "")).lower() in {"valid", "oracle_level_preferred", "improved_candidate_not_oracle"}]
    if not valid:
        raise BootstrapError("No valid MedSAM releases available in generated mask registry")
    valid.sort(key=lambda r: str(r.get("created_at", "")))
    return str(valid[-1]["mask_release_id"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_select_latest_valid_release_when_not_pinned tests/unit/test_labelstudio_bootstrap.py::test_fail_when_pinned_release_missing -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/label_studio_medsam_hybrid.yaml src/eq/labelstudio/bootstrap.py tests/unit/test_labelstudio_bootstrap.py
git commit -m "Add YAML-driven MedSAM release selection for Label Studio"
```

---

### Task 2: CLI Positional Image Directory + Config Wiring

**Files:**
- Modify: `src/eq/__main__.py`
- Modify: `src/eq/labelstudio/bootstrap.py`
- Test: `tests/unit/test_labelstudio_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_labelstudio_cli_accepts_positional_image_dir(monkeypatch, tmp_path, capsys):
    import eq.__main__ as main
    captured = {}

    def fake_run_bootstrap(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message="ready",
            project_url="",
            task_manifest_path=tmp_path / "tasks.json",
            plan=SimpleNamespace(url="http://localhost:8080"),
        )

    monkeypatch.setattr(main, "_load_labelstudio_bootstrap", lambda: fake_run_bootstrap)
    args = SimpleNamespace(
        image_dir=str(tmp_path / "images"),
        images=None,
        config=None,
        project_name="EQ",
        runtime_root=None,
        port=8080,
        container_name="eq-labelstudio",
        docker_image="heartexlabs/label-studio:latest",
        username="eq-admin@example.local",
        password="eq-labelstudio",
        api_token="eq-local-token",
        timeout_seconds=60,
        dry_run=True,
    )
    main.labelstudio_start_command(args)
    assert captured["images_dir"] == tmp_path / "images"
```

- [ ] **Step 2: Run test to verify it fails**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_labelstudio_cli_accepts_positional_image_dir -q`

Expected: FAIL because parser/command currently expects only `--images`.

- [ ] **Step 3: Write minimal implementation**

```python
# in argparse setup for labelstudio start:
labelstudio_start_parser.add_argument("image_dir", nargs="?", help="Image directory")
labelstudio_start_parser.add_argument("--images", help="Legacy image directory flag")
labelstudio_start_parser.add_argument("--config", help="Hybrid Label Studio config YAML")

# in labelstudio_start_command:
raw_images = args.image_dir or args.images
if not raw_images:
    raise ValueError("Provide image directory as positional argument or --images")
result = run_bootstrap(images_dir=Path(raw_images), config_path=Path(args.config) if args.config else None, ...)
```

- [ ] **Step 4: Run tests to verify it passes**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_labelstudio_cli_accepts_positional_image_dir -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eq/__main__.py src/eq/labelstudio/bootstrap.py tests/unit/test_labelstudio_bootstrap.py
git commit -m "Support positional image-dir for labelstudio start"
```

---

### Task 3: Preload Prediction Payload + Companion Health Gate

**Files:**
- Modify: `src/eq/labelstudio/bootstrap.py`
- Test: `tests/unit/test_labelstudio_bootstrap.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_bootstrap_builds_predictions_from_release_manifest(tmp_path):
    rows = [{"source_relative_path": "animal_1/a.png", "mask_release_id": "r3", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
    tasks = [{"data": {"source_relative_path": "animal_1/a.png", "image": "/data/local-files/?d=animal_1/a.png"}}]
    enriched = _attach_release_predictions(tasks, rows, "r3")
    assert "predictions" in enriched[0]
    assert enriched[0]["predictions"][0]["model_version"] == "medsam:r3"


def test_companion_health_required_fails_closed():
    with pytest.raises(BootstrapError, match="box-assisted companion health check failed"):
        _ensure_companion_ready("http://localhost:9999", require=True, allow_offline_manual_only=False, timeout_seconds=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_bootstrap_builds_predictions_from_release_manifest tests/unit/test_labelstudio_bootstrap.py::test_companion_health_required_fails_closed -q`

Expected: FAIL due to missing helpers/behavior.

- [ ] **Step 3: Write minimal implementation**

```python
def _ensure_companion_ready(base_url: str, require: bool, allow_offline_manual_only: bool, timeout_seconds: int) -> None:
    if not require:
        return
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/healthz", method="GET")
        with urllib.request.urlopen(req, timeout=min(5, timeout_seconds)) as resp:
            if resp.status != 200:
                raise BootstrapError("box-assisted companion health check failed")
    except Exception as exc:
        if allow_offline_manual_only:
            return
        raise BootstrapError("box-assisted companion health check failed") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py::test_bootstrap_builds_predictions_from_release_manifest tests/unit/test_labelstudio_bootstrap.py::test_companion_health_required_fails_closed -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eq/labelstudio/bootstrap.py tests/unit/test_labelstudio_bootstrap.py
git commit -m "Add MedSAM preload predictions and companion health gating"
```

---

### Task 4: Export Lineage Fields + Validator Hard-Fail Rules

**Files:**
- Modify: `src/eq/labelstudio/glomerulus_grading.py`
- Create: `tests/fixtures/labelstudio_glomerulus_instances/hybrid_auto_edited_export.json`
- Create: `tests/fixtures/labelstudio_glomerulus_instances/hybrid_box_assisted_export.json`
- Create: `tests/fixtures/labelstudio_glomerulus_instances/hybrid_contradictory_lineage_export.json`
- Modify: `tests/unit/test_labelstudio_glomerulus_grading.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_hybrid_auto_edited_lineage_fields_present():
    records = load_glomerulus_grading_records("tests/fixtures/labelstudio_glomerulus_instances/hybrid_auto_edited_export.json")
    row = records.iloc[0]
    assert row["proposal_kind"] == "auto_preload"
    assert row["region_edit_state"] == "human_refined_boundary"
    assert row["mask_release_id"] == "r3"


def test_hybrid_contradictory_lineage_fails():
    with pytest.raises(LabelStudioGlomerulusContractError, match="contradictory lineage"):
        load_glomerulus_grading_records("tests/fixtures/labelstudio_glomerulus_instances/hybrid_contradictory_lineage_export.json")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_glomerulus_grading.py::test_hybrid_auto_edited_lineage_fields_present tests/unit/test_labelstudio_glomerulus_grading.py::test_hybrid_contradictory_lineage_fails -q`

Expected: FAIL due to missing lineage columns/validation branch.

- [ ] **Step 3: Write minimal implementation**

```python
# extend emitted record fields
record.update(
    {
        "mask_release_id": lineage.get("mask_release_id"),
        "proposal_kind": lineage.get("proposal_kind"),
        "region_edit_state": lineage.get("region_edit_state"),
    }
)

# fail-closed contradictory lineage
if record["proposal_kind"] == "auto_preload" and not record["mask_release_id"]:
    raise LabelStudioGlomerulusContractError("contradictory lineage: auto_preload requires mask_release_id")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_glomerulus_grading.py::test_hybrid_auto_edited_lineage_fields_present tests/unit/test_labelstudio_glomerulus_grading.py::test_hybrid_contradictory_lineage_fails -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eq/labelstudio/glomerulus_grading.py tests/unit/test_labelstudio_glomerulus_grading.py tests/fixtures/labelstudio_glomerulus_instances/hybrid_*.json
git commit -m "Add hybrid MedSAM lineage export fields and strict validation"
```

---

### Task 5: Documentation + Final Validation Gate

**Files:**
- Modify: `README.md`
- Modify: `docs/LABEL_STUDIO_GLOMERULUS_GRADING.md`
- Modify: `openspec/changes/label-studio-medsam-hybrid-grading/tasks.md` (check boxes as executed)

- [ ] **Step 1: Write docs tests (failing expectations via grep assertions in test)**

```python
def test_docs_reference_positional_labelstudio_command():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "eq labelstudio start <image-dir>" in text
```

- [ ] **Step 2: Run test to verify it fails (if docs not updated yet)**

Run:  
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_docs_smoke.py::test_docs_reference_positional_labelstudio_command -q`

Expected: FAIL until docs updated.

- [ ] **Step 3: Update docs with minimal collaborator path + YAML pointer**

```markdown
eq labelstudio start /path/to/images

# Advanced settings:
# configs/label_studio_medsam_hybrid.yaml
```

- [ ] **Step 4: Run full validation suite**

Run:
`openspec validate label-studio-medsam-hybrid-grading --strict`  
Expected: `Change 'label-studio-medsam-hybrid-grading' is valid`

Run:
`python3 scripts/check_openspec_explicitness.py label-studio-medsam-hybrid-grading`  
Expected: `OpenSpec explicitness check passed for label-studio-medsam-hybrid-grading`

Run:
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/unit/test_labelstudio_bootstrap.py tests/unit/test_labelstudio_glomerulus_grading.py -q`  
Expected: PASS

Run:
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/__main__.py src/eq/labelstudio/bootstrap.py src/eq/labelstudio/glomerulus_grading.py tests/unit/test_labelstudio_*.py`  
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add README.md docs/LABEL_STUDIO_GLOMERULUS_GRADING.md openspec/changes/label-studio-medsam-hybrid-grading/tasks.md
git commit -m "Document hybrid MedSAM Label Studio workflow and finalize validation"
```

---

## Self-Review Notes (completed)

- Spec coverage: plan tasks cover CLI minimization, YAML-first config, latest valid release audit/selection, preload integration, companion health gate, lineage export, and strict validation.
- Placeholder scan: removed TBD/TODO style placeholders; each step has explicit commands/code.
- Type consistency: uses one release id term (`mask_release_id`) and one provenance key naming set (`proposal_kind`, `region_edit_state`) across tasks.
