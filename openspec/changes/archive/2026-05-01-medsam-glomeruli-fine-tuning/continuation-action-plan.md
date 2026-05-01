# MedSAM Fine-Tuned Checkpoint Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete fine-tuned checkpoint evaluation on fixed validation/test examples, produce gate decisions, and update OpenSpec notes with an evidence-based recommendation.

**Architecture:** Reuse existing fixed-split baseline owners and the new checkpoint provenance owner, then add a focused fine-tuned inference/evaluation pass that writes metrics/overlays under the existing fine-tuning run directory. Keep the run contract deterministic by using the same split manifests and metric schema already used for automatic/oracle/current/trivial baselines.

**Tech Stack:** Python 3.10 (`eq-mac`), PyTorch MPS, pandas, existing `eq.evaluation` workflow helpers, OpenSpec change artifacts.

---

## File Structure / Ownership

- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
  - Owner for fixed-split orchestration, MedSAM batch invocation, checkpoint provenance, and summary fields.
- Modify: `configs/medsam_glomeruli_fine_tuning.yaml`
  - Owner for run toggles and thresholds used by fine-tuned evaluation.
- Modify: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`
  - Owner for unit tests for new fine-tuned evaluation outputs and decision fields.
- Modify: `openspec/changes/medsam-glomeruli-fine-tuning/tasks.md`
  - Track task completion state.
- Modify: `openspec/changes/medsam-glomeruli-fine-tuning/implementation-notes.md`
  - Record pilot evidence and final recommendation.

## Scope Check

This continuation is one subsystem (fine-tuned checkpoint evaluation + decision packaging) and can ship as one plan.

---

### Task 1: Add Fine-Tuned Fixed-Split Inference/Evaluation Outputs

**Files:**

- Modify: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- **Step 1: Write the failing test**

```python
def test_finetuned_evaluation_summary_fields_present_when_checkpoint_exists(tmp_path: Path):
    summary = {
        "finetuned_evaluation": {
            "status": "completed",
            "metric_rows": 42,
            "metrics_csv": "x.csv",
            "overlays_dir": "overlays",
            "prompt_failure_count": 0,
        }
    }
    assert summary["finetuned_evaluation"]["status"] == "completed"
    assert summary["finetuned_evaluation"]["metric_rows"] > 0
```

- **Step 2: Run test to verify it fails**

Run: `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_medsam_glomeruli_fine_tuning_workflow.py::test_finetuned_evaluation_summary_fields_present_when_checkpoint_exists -v`
Expected: FAIL because workflow/test scaffolding for `finetuned_evaluation` is incomplete.

- **Step 3: Write minimal implementation**

```python
# run_medsam_glomeruli_fine_tuning_workflow.py (conceptual target)
if checkpoint_provenance["supported_checkpoint"]:
    finetuned_summary = _run_finetuned_fixed_split_evaluation(...)
else:
    finetuned_summary = {"status": "skipped_missing_supported_checkpoint"}

summary["finetuned_evaluation"] = finetuned_summary
```

- **Step 4: Run test to verify it passes**

Run: `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_medsam_glomeruli_fine_tuning_workflow.py::test_finetuned_evaluation_summary_fields_present_when_checkpoint_exists -v`
Expected: PASS

- **Step 5: Commit**

```bash
git add tests/test_medsam_glomeruli_fine_tuning_workflow.py src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py
git commit -m "feat: add fine-tuned fixed-split evaluation summary outputs"
```

---

### Task 2: Add Gate Comparison Against Oracle/Auto/Current/Trivial Baselines

**Files:**

- Modify: `tests/test_medsam_glomeruli_fine_tuning_workflow.py`
- Modify: `src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py`
- **Step 1: Write the failing test**

```python
def test_finetuned_comparison_includes_oracle_gap_and_adoption_tier():
    decision = {
        "oracle_dice_gap": 0.03,
        "adoption_tier": "oracle_level_preferred",
        "improves_current_auto": True,
        "improves_current_segmenter": True,
        "beats_trivial_baseline": True,
    }
    assert decision["oracle_dice_gap"] <= 0.05
    assert decision["adoption_tier"] in {
        "oracle_level_preferred",
        "improved_candidate_not_oracle",
        "blocked",
    }
```

- **Step 2: Run test to verify it fails**

Run: `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_medsam_glomeruli_fine_tuning_workflow.py::test_finetuned_comparison_includes_oracle_gap_and_adoption_tier -v`
Expected: FAIL before comparison summary is fully wired.

- **Step 3: Write minimal implementation**

```python
# run_medsam_glomeruli_fine_tuning_workflow.py (conceptual target)
comparison = _classify_generated_mask_adoption(
    fine_tuned_metrics=finetuned_mean,
    oracle_metrics=oracle_mean,
    current_auto_metrics=auto_mean,
    current_segmenter_metrics=current_best_mean,
    trivial_baseline_metrics=trivial_mean,
    prompt_failure_count=finetuned_prompt_failures,
    gates=_mapping(config, "adoption_gates"),
    overlay_review_status=overlay_review_status,
)
summary["finetuned_comparison"] = comparison
```

- **Step 4: Run test to verify it passes**

Run: `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_medsam_glomeruli_fine_tuning_workflow.py::test_finetuned_comparison_includes_oracle_gap_and_adoption_tier -v`
Expected: PASS

- **Step 5: Commit**

```bash
git add tests/test_medsam_glomeruli_fine_tuning_workflow.py src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py
git commit -m "feat: add fine-tuned vs baseline gate comparison summary"
```

---

### Task 3: Run Full Pilot Evaluation and Validate Outputs

**Files:**

- Modify: `configs/medsam_glomeruli_fine_tuning.yaml` (if toggles needed)
- Modify: `openspec/changes/medsam-glomeruli-fine-tuning/tasks.md`
- Modify: `openspec/changes/medsam-glomeruli-fine-tuning/implementation-notes.md`
- **Step 1: Run the full workflow on runtime data**

Run:
`EQ_RUNTIME_ROOT=/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier PYTORCH_ENABLE_MPS_FALLBACK=1 MPLCONFIGDIR=/tmp/mpl_eq /Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m eq run-config --config configs/medsam_glomeruli_fine_tuning.yaml`

Expected: command exits 0; summary contains `finetuned_evaluation` and `finetuned_comparison`.

- **Step 2: Verify required output artifacts exist**

Run:
`/Users/ncamarda/mambaforge/envs/eq-mac/bin/python - <<'PY' from pathlib import Path root = Path('/Users/ncamarda/ProjectsRuntime/endotheliosis_quantifier/output/segmentation_evaluation/medsam_glomeruli_fine_tuning/pilot_medsam_glomeruli_fine_tuning') required = [     root / 'summary.json',     root / 'baseline_metrics' / 'automatic_medsam_metrics.csv',     root / 'baseline_metrics' / 'oracle_medsam_metrics.csv',     root / 'baseline_metrics' / 'current_segmenter_metrics.csv', ] print(all(p.exists() for p in required), [str(p) for p in required if not p.exists()]) PY`

Expected: `True []`

- **Step 3: Validate focused tests/lint/spec**

Run:

- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m pytest tests/test_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_torch_runtime.py -q`
- `/Users/ncamarda/mambaforge/envs/eq-mac/bin/python -m ruff check src/eq/evaluation/run_medsam_glomeruli_fine_tuning_workflow.py tests/test_medsam_glomeruli_fine_tuning_workflow.py`
- `openspec validate medsam-glomeruli-fine-tuning --strict`

Expected: all pass.

- **Step 4: Update OpenSpec task state**

```markdown
- [x] 5.1 Run automatic proposal-box MedSAM inference with fine-tuned checkpoints on fixed validation/test examples.
- [x] 5.2 Write fine-tuned masks, prompt provenance, metrics, overlays, and prompt failures under output root.
- [x] 5.3 Compare fine-tuned checkpoints against oracle/automatic/current/trivial baselines.
- [x] 7.4 Review summary, metrics, prompt failures, checkpoint provenance, overlays, and gate fields.
- [x] 7.5 Record pilot decision and recommendation in implementation notes.
```

- **Step 5: Commit**

```bash
git add openspec/changes/medsam-glomeruli-fine-tuning/tasks.md openspec/changes/medsam-glomeruli-fine-tuning/implementation-notes.md
git commit -m "docs: record fine-tuned pilot decision and next recommendation"
```

---

### Task 4: Decide Promotion Path (No Silent Rollout)

**Files:**

- Modify: `openspec/changes/medsam-glomeruli-fine-tuning/implementation-notes.md`
- **Step 1: Write explicit decision block**

```markdown
## Pilot Decision
- adoption_tier: <oracle_level_preferred|improved_candidate_not_oracle|blocked>
- recommended_generated_mask_source: <value>
- rationale: <1-2 sentences tied to oracle gap and reliability gates>
```

- **Step 2: Add operational recommendation**

```markdown
## Next Recommendation
1. Keep current automatic source (or transition to fine-tuned release) for downstream opt-in only.
2. Require overlay review sign-off before any release packaging.
3. Do not update README/model-status claims until downstream grading stability review.
```

- **Step 3: Validate docs/spec**

Run: `openspec validate medsam-glomeruli-fine-tuning --strict`
Expected: valid.

- **Step 4: Commit**

```bash
git add openspec/changes/medsam-glomeruli-fine-tuning/implementation-notes.md
git commit -m "docs: finalize pilot decision and rollout recommendation"
```

---

## Self-Review

- Spec coverage:
  - Remaining implementation surfaces in tasks 5.1–5.3 and 7.4–7.5 are covered by Tasks 1–4 above.
  - Previously completed checkpoint and baseline work is acknowledged; no duplicate implementation tasks added.
- Placeholder scan:
  - No `TBD/TODO/implement later` placeholders remain.
  - Each command step includes exact run commands and expected outcomes.
- Type/contract consistency:
  - Uses existing adoption-tier values (`oracle_level_preferred`, `improved_candidate_not_oracle`, `blocked`) and existing summary keys/contracts already present in the workflow.

