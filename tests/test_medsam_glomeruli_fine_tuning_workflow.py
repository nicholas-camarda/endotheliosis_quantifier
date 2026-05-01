import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow import (
    _baseline_metric_rows,
    _build_adaptation_command,
    _build_checkpoint_provenance,
    _build_deterministic_splits,
    _build_finetuned_comparison_summary,
    _classify_generated_mask_adoption,
    _package_generated_mask_release,
    _preflight_training_dependencies,
    _prepare_medsam_npy_data,
    _run_external_fixed_split_baselines,
    _run_finetuned_fixed_split_evaluation,
    _training_command_for_config,
    _trivial_baseline_aggregate,
    _update_generated_mask_registry,
    _validate_split_manifest,
    run_medsam_glomeruli_fine_tuning_workflow,
)
from eq.utils.paths import ensure_not_under_runtime_raw_data


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(path)


def test_fixed_split_generation_keeps_source_groups_together():
    rows = pd.DataFrame(
        [
            {
                "manifest_row_id": f"row-{idx}",
                "cohort_id": "vegfri_dox",
                "lane_assignment": "manual_mask_external",
                "source_sample_id": group,
                "split_group_id": group,
                "image_path": f"image-{idx}.png",
                "mask_path": f"mask-{idx}.png",
            }
            for idx, group in enumerate(["a", "a", "b", "c", "d", "e"], start=1)
        ]
    )

    split = _build_deterministic_splits(
        rows, train_fraction=0.5, validation_fraction=0.25
    )

    assert split.groupby("split_group_id")["split"].nunique().max() == 1
    assert set(split["split"]) == {"train", "validation", "test"}
    assert "selection_rank" in split.columns
    assert "selection_reason" in split.columns


def test_explicit_split_manifest_validation_hashes_required_paths(tmp_path: Path):
    runtime_root = tmp_path
    image = np.zeros((6, 6), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    _write_png(runtime_root / "raw_data/cohorts/demo/images/img.png", image)
    _write_png(runtime_root / "raw_data/cohorts/demo/masks/mask.png", mask)
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "manifest_row_id": "row-1",
                "cohort_id": "demo",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "sample-1",
                "split_group_id": "sample-1",
                "image_path": "raw_data/cohorts/demo/images/img.png",
                "mask_path": "raw_data/cohorts/demo/masks/mask.png",
                "split": "validation",
                "selection_rank": 1,
                "selection_reason": "test",
            }
        ]
    ).to_csv(split_path, index=False)

    digest = _validate_split_manifest(split_path, runtime_root)

    assert len(digest) == 64


def test_generated_mask_release_rejects_raw_data_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="raw_data"):
        ensure_not_under_runtime_raw_data(
            tmp_path / "raw_data/cohorts/demo/masks/generated", tmp_path
        )


def test_trivial_baseline_rows_have_metric_schema():
    manual = np.zeros((4, 4), dtype=np.uint8)
    manual[1:3, 1:3] = 1

    rows = _baseline_metric_rows(
        manual_mask=manual,
        manifest_row_id="row-1",
        cohort_id="demo",
        lane_assignment="manual_mask_core",
    )

    assert {row["method"] for row in rows} == {
        "trivial_all_background",
        "trivial_all_foreground",
    }
    assert all("dice" in row and "jaccard" in row for row in rows)
    assert rows[0]["candidate_artifact"] == "trivial_baseline"


def test_trivial_baseline_aggregate_fail_closed_for_empty_csv(tmp_path: Path) -> None:
    grouped_csv = tmp_path / "trivial_grouped.csv"
    grouped_csv.write_text("", encoding="utf-8")

    aggregate = _trivial_baseline_aggregate(
        {"trivial_baseline_metric_by_source": str(grouped_csv)}
    )

    assert aggregate == {"dice": 0.0, "jaccard": 0.0}


def test_trivial_baseline_aggregate_fail_closed_for_missing_metric_columns(
    tmp_path: Path,
) -> None:
    grouped_csv = tmp_path / "trivial_grouped.csv"
    pd.DataFrame([{"source_sample_id": "s1", "foo": "1.0"}]).to_csv(
        grouped_csv, index=False
    )

    aggregate = _trivial_baseline_aggregate(
        {"trivial_baseline_metric_by_source": str(grouped_csv)}
    )

    assert aggregate == {"dice": 0.0, "jaccard": 0.0}


def test_trivial_baseline_aggregate_fail_closed_for_malformed_csv(tmp_path: Path) -> None:
    grouped_csv = tmp_path / "trivial_grouped.csv"
    grouped_csv.write_text('dice,jaccard\n"0.5,0.4\n', encoding="utf-8")

    aggregate = _trivial_baseline_aggregate(
        {"trivial_baseline_metric_by_source": str(grouped_csv)}
    )

    assert aggregate == {"dice": 0.0, "jaccard": 0.0}


def test_adoption_gate_distinguishes_oracle_level_candidate_and_blocked():
    oracle = _classify_generated_mask_adoption(
        fine_tuned_metrics={"dice": 0.91, "jaccard": 0.84},
        oracle_metrics={"dice": 0.94, "jaccard": 0.86},
        current_auto_metrics={"dice": 0.78, "jaccard": 0.64},
        current_segmenter_metrics={"dice": 0.72, "jaccard": 0.58},
        trivial_baseline_metrics={"dice": 0.20, "jaccard": 0.12},
        prompt_failure_count=0,
        gates={"min_dice": 0.90, "min_jaccard": 0.82, "max_oracle_dice_gap": 0.05},
        overlay_review_status="passed",
    )
    assert oracle["adoption_tier"] == "oracle_level_preferred"
    assert oracle["recommended_generated_mask_source"] == "medsam_finetuned_glomeruli"
    assert oracle["oracle_dice_gap"] == pytest.approx(0.03)

    improved = _classify_generated_mask_adoption(
        fine_tuned_metrics={"dice": 0.86, "jaccard": 0.76},
        oracle_metrics={"dice": 0.94, "jaccard": 0.86},
        current_auto_metrics={"dice": 0.78, "jaccard": 0.64},
        current_segmenter_metrics={"dice": 0.72, "jaccard": 0.58},
        trivial_baseline_metrics={"dice": 0.20, "jaccard": 0.12},
        prompt_failure_count=0,
        gates={"min_dice": 0.90, "min_jaccard": 0.82, "max_oracle_dice_gap": 0.05},
        overlay_review_status="passed",
    )
    assert improved["adoption_tier"] == "improved_candidate_not_oracle"
    assert improved["recommended_generated_mask_source"] == ""

    blocked = _classify_generated_mask_adoption(
        fine_tuned_metrics={"dice": 0.75, "jaccard": 0.60},
        oracle_metrics={"dice": 0.94, "jaccard": 0.86},
        current_auto_metrics={"dice": 0.78, "jaccard": 0.64},
        current_segmenter_metrics={"dice": 0.72, "jaccard": 0.58},
        trivial_baseline_metrics={"dice": 0.20, "jaccard": 0.12},
        prompt_failure_count=0,
        gates={"min_dice": 0.90, "min_jaccard": 0.82, "max_oracle_dice_gap": 0.05},
        overlay_review_status="passed",
    )
    assert blocked["adoption_tier"] == "blocked"
    assert blocked["failure_mode"] == "oracle_gap"


def test_generated_mask_release_and_registry_include_required_fields(tmp_path: Path):
    run_mask = tmp_path / "run/masks/row-1.png"
    _write_png(run_mask, np.ones((4, 4), dtype=np.uint8))
    release_dir = tmp_path / "derived_data/generated_masks/glomeruli/medsam_finetuned/release-1"
    registry = tmp_path / "derived_data/generated_masks/glomeruli/manifest.csv"

    release = _package_generated_mask_release(
        release_dir=release_dir,
        mask_release_id="release-1",
        run_id="run-1",
        checkpoint_id="ckpt-1",
        checkpoint_path=tmp_path / "models/medsam_glomeruli/ckpt-1",
        proposal_source="transfer",
        proposal_threshold=0.35,
        adoption_tier="improved_candidate_not_oracle",
        generated_masks=[
            {
                "manifest_row_id": "row-1",
                "cohort_id": "demo",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "sample-1",
                "source_image_path": "raw_data/cohorts/demo/images/img.png",
                "reference_mask_path": "raw_data/cohorts/demo/masks/mask.png",
                "generated_mask_path": str(run_mask),
                "generation_status": "generated",
            }
        ],
    )
    _update_generated_mask_registry(registry, release["manifest"])

    manifest = pd.read_csv(release["manifest"])
    registry_frame = pd.read_csv(registry)
    assert manifest.loc[0, "mask_source"] == "medsam_finetuned_glomeruli"
    assert manifest.loc[0, "adoption_tier"] == "improved_candidate_not_oracle"
    assert registry_frame.loc[0, "release_manifest_path"] == str(release["manifest"])
    assert (release_dir / "INDEX.md").exists()
    assert (release_dir / "provenance.json").exists()


def test_preflight_and_provenance_fail_closed_without_success_status(tmp_path: Path):
    config = {
        "medsam": {
            "python": str(tmp_path / "missing-python"),
            "repo": str(tmp_path / "missing-repo"),
            "base_checkpoint": str(tmp_path / "missing.pth"),
            "adaptation_mode": "frozen_image_encoder_mask_decoder",
            "frozen_components": ["image_encoder", "prompt_encoder"],
            "trainable_components": ["mask_decoder"],
        },
        "training": {"entrypoint": "train_one_gpu.py"},
    }
    with pytest.raises(FileNotFoundError, match="MedSAM Python interpreter"):
        _preflight_training_dependencies(config)

    provenance = _build_checkpoint_provenance(
        training_command=["python", "train_one_gpu.py"],
        environment={"PYTHON": "python"},
        medsam_repo=tmp_path / "MedSAM",
        base_checkpoint=tmp_path / "base.pth",
        split_manifest_paths={"train": tmp_path / "train.csv"},
        split_manifest_hashes={"train": "abc"},
        adaptation_mode="frozen_image_encoder_mask_decoder",
        frozen_components=["image_encoder", "prompt_encoder"],
        trainable_components=["mask_decoder"],
        hyperparameters={"epochs": 1},
        checkpoint_files=[],
        training_status="failed_dependency_preflight",
        local_feasibility_status="requires_external_accelerator",
    )

    assert provenance["training_status"] == "failed_dependency_preflight"
    assert provenance["checkpoint_files"] == []
    assert provenance["supported_checkpoint"] is False


def test_adaptation_command_wraps_upstream_training_without_vendoring(tmp_path: Path):
    command = _build_adaptation_command(
        medsam_python=tmp_path / "env/bin/python",
        entrypoint=tmp_path / "MedSAM/train_one_gpu.py",
        train_npy_root=tmp_path / "npy_data",
        work_dir=tmp_path / "models/medsam_glomeruli/ckpt-1",
        base_checkpoint=tmp_path / "medsam_vit_b.pth",
        task_name="eq_glomeruli",
        epochs=3,
        batch_size=1,
        learning_rate=0.0001,
        device="mps",
    )

    assert command[:2] == [
        str(tmp_path / "env/bin/python"),
        str(tmp_path / "MedSAM/train_one_gpu.py"),
    ]
    assert "-i" in command
    assert "-pretrain_model_path" in command
    assert "mps" in command


def test_training_command_eq_native_adapter_invokes_in_repo_adapter(tmp_path: Path) -> None:
    config = {"medsam": {"device": "mps"}}
    preflight = {
        "medsam_python": str(tmp_path / "py"),
        "medsam_repo": str(tmp_path / "repo"),
        "base_checkpoint": str(tmp_path / "ckpt.pth"),
        "training_backend": "eq_native_adapter",
        "entrypoint": "",
    }
    (tmp_path / "py").write_bytes(b"")
    cmd = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=tmp_path / "npy",
        work_dir=tmp_path / "wd",
        training_cfg={"epochs": 2, "batch_size": 1},
    )
    assert cmd[1] == "-m"
    assert "eq.evaluation.medsam_glomeruli_adapter" in cmd
    assert "--lr-scheduler" in cmd
    assert cmd[cmd.index("--lr-scheduler") + 1] == "none"


def test_training_command_eq_native_adapter_passes_val_npy_when_set(tmp_path: Path) -> None:
    config = {"medsam": {"device": "mps"}}
    preflight = {
        "medsam_python": str(tmp_path / "py"),
        "medsam_repo": str(tmp_path / "repo"),
        "base_checkpoint": str(tmp_path / "ckpt.pth"),
        "training_backend": "eq_native_adapter",
        "entrypoint": "",
    }
    (tmp_path / "py").write_bytes(b"")
    val_root = tmp_path / "val_npy"
    cmd = _training_command_for_config(
        config=config,
        preflight=preflight,
        train_npy_root=tmp_path / "npy",
        work_dir=tmp_path / "wd",
        training_cfg={"epochs": 2, "batch_size": 1, "val_max_examples": 8},
        val_npy_root=val_root,
    )
    i = cmd.index("--val-npy-root")
    assert cmd[i + 1] == str(val_root)
    j = cmd.index("--val-max-examples")
    assert cmd[j + 1] == "8"


def test_prepare_medsam_npy_data_writes_paired_normalized_arrays(tmp_path: Path):
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[..., 0] = 128
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    image_path = tmp_path / "raw_data/cohorts/demo/images/img.png"
    mask_path = tmp_path / "raw_data/cohorts/demo/masks/mask.png"
    _write_png(image_path, image[..., 0])
    Image.fromarray(image).save(image_path)
    _write_png(mask_path, mask)
    split = pd.DataFrame(
        [
            {
                "manifest_row_id": "row-1",
                "cohort_id": "demo",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "sample-1",
                "split_group_id": "sample-1",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "split": "train",
                "selection_rank": 1,
                "selection_reason": "test",
            }
        ]
    )

    summary = _prepare_medsam_npy_data(
        train_rows=split,
        output_root=tmp_path / "derived_data/medsam_glomeruli/npy_data",
        image_size=32,
    )

    img = np.load(summary["image_files"][0])
    gt = np.load(summary["mask_files"][0])
    assert summary["prepared_count"] == 1
    assert img.shape == (32, 32, 3)
    assert gt.shape == (32, 32)
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert set(np.unique(gt).tolist()) == {0, 1}


def test_external_fixed_split_baselines_skips_when_fixed_rows_empty(tmp_path: Path):
    cfg = {
        "baseline_evaluation": {},
        "proposal": {},
        "tiling": {},
        "current_segmenter": {},
        "medsam": {},
    }
    out = _run_external_fixed_split_baselines(
        fixed_rows=pd.DataFrame(),
        config=cfg,
        baseline_dir=tmp_path / "baseline_metrics",
        _runtime_root=tmp_path,
    )
    assert out["status"] == "skipped_empty_fixed_split"


def test_external_fixed_split_oracle_only_invokes_medsam_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "MedSAM"
    repo.mkdir()
    (repo / "MedSAM_Inference.py").write_text("# stub\n", encoding="utf-8")
    fake_python = tmp_path / "fake_python"
    fake_python.write_bytes(b"")
    checkpoint = tmp_path / "medsam.pth"
    checkpoint.write_bytes(b"")

    cfg = {
        "baseline_evaluation": {
            "external_baselines": True,
            "automatic_medsam": False,
            "oracle_prompt_medsam": True,
            "current_segmenter": False,
        },
        "proposal": {"min_component_area": 1, "padding": 0},
        "tiling": {},
        "current_segmenter": {},
        "medsam": {
            "python": str(fake_python),
            "repo": str(repo),
            "base_checkpoint": str(checkpoint),
            "device": "cpu",
        },
    }

    image_path = tmp_path / "img.png"
    mask_path = tmp_path / "msk.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    _write_png(mask_path, mask)

    fixed_rows = pd.DataFrame(
        [
            {
                "manifest_row_id": "row-1",
                "cohort_id": "c1",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "s1",
                "image_path": "img.png",
                "mask_path": "msk.png",
                "image_path_resolved": str(image_path),
                "mask_path_resolved": str(mask_path),
                "selection_rank": 1,
                "selection_reason": "test",
            }
        ]
    )

    calls: list = []

    def fake_batch(**kwargs):
        calls.append(kwargs)
        for item in kwargs["items"]:
            Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((16, 16), dtype=np.uint8) * 255).save(
                item["output_path"]
            )
        return []

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch",
        fake_batch,
    )

    out = _run_external_fixed_split_baselines(
        fixed_rows=fixed_rows,
        config=cfg,
        baseline_dir=tmp_path / "baseline_metrics",
        _runtime_root=tmp_path,
    )

    assert len(calls) == 1
    assert out["status"] == "completed"
    assert "oracle_medsam_metrics_csv" in out


def test_workflow_summary_marks_finetuned_evaluation_completed_when_real_inference_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path
    image_path = runtime_root / "raw_data/cohorts/demo/images/img.png"
    mask_path = runtime_root / "raw_data/cohorts/demo/masks/mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    _write_png(mask_path, mask)

    split_row = {
        "manifest_row_id": "row-1",
        "cohort_id": "demo",
        "lane_assignment": "manual_mask_core",
        "source_sample_id": "sample-1",
        "split_group_id": "sample-1",
        "image_path": "raw_data/cohorts/demo/images/img.png",
        "mask_path": "raw_data/cohorts/demo/masks/mask.png",
        "split": "train",
        "selection_rank": 1,
        "selection_reason": "test",
    }
    split_dir = runtime_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame([{**split_row, "split": "validation"}]).to_csv(
        split_dir / "validation.csv", index=False
    )
    pd.DataFrame([{**split_row, "split": "test"}]).to_csv(
        split_dir / "test.csv", index=False
    )

    manifest_path = runtime_root / "raw_data/cohorts/manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / "models/medsam/base.pth"
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b"base")
    medsam_repo = runtime_root / "vendor/MedSAM"
    medsam_repo.mkdir(parents=True, exist_ok=True)
    (medsam_repo / "MedSAM_Inference.py").write_text("# stub\n", encoding="utf-8")
    medsam_python = runtime_root / "bin/python"
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    checkpoint_root = runtime_root / "models/medsam_glomeruli/test-run"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "finetuned_epoch_1.pth").write_bytes(b"finetuned")
    transfer_model = runtime_root / "models/current/transfer.pkl"
    scratch_model = runtime_root / "models/current/scratch.pkl"
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b"transfer")
    scratch_model.write_bytes(b"scratch")

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        "eq.data_management.model_loading.load_model_safely",
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability",
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )

    def _fake_batch(**kwargs):
        for item in kwargs["items"]:
            Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((12, 12), dtype=np.uint8) * 255).save(
                item["output_path"]
            )
        failures_path = Path(kwargs["output_dir"]) / "medsam_batch_failures.json"
        failures_path.write_text("[]", encoding="utf-8")
        return []

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch",
        _fake_batch,
    )

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies",
        lambda _cfg: {
            "medsam_python": str(medsam_python),
            "medsam_repo": str(medsam_repo),
            "base_checkpoint": str(base_checkpoint),
            "entrypoint": "",
            "adaptation_mode": "frozen_image_encoder_mask_decoder",
            "training_backend": "eq_native_adapter",
        },
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config",
        lambda **_kwargs: ["python", "-m", "eq.evaluation.medsam_glomeruli_adapter"],
    )

    config_path = runtime_root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "workflow: medsam_glomeruli_fine_tuning",
                "run:",
                "  name: test-run",
                f"  runtime_root_default: {runtime_root}",
                "medsam:",
                f"  python: {medsam_python}",
                f"  repo: {medsam_repo}",
                f"  base_checkpoint: {base_checkpoint}",
                "  device: cpu",
                "  adaptation_mode: frozen_image_encoder_mask_decoder",
                "  frozen_components: [image_encoder, prompt_encoder]",
                "  trainable_components: [mask_decoder]",
                "inputs:",
                f"  manifest_path: {manifest_path.relative_to(runtime_root)}",
                "  lane_assignments: [manual_mask_core]",
                "  split_manifests:",
                f"    train: { (split_dir / 'train.csv').relative_to(runtime_root) }",
                f"    validation: { (split_dir / 'validation.csv').relative_to(runtime_root) }",
                f"    test: { (split_dir / 'test.csv').relative_to(runtime_root) }",
                "outputs:",
                "  evaluation_dir: output/test-eval",
                f"  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}",
                f"  generated_mask_release_root: {(runtime_root / 'derived_data/generated_masks/glomeruli/medsam_finetuned/test-run').relative_to(runtime_root)}",
                f"  generated_mask_registry_path: {(runtime_root / 'derived_data/generated_masks/glomeruli/manifest.csv').relative_to(runtime_root)}",
                "baseline_evaluation:",
                "  external_baselines: false",
                "current_segmenter:",
                f"  transfer_model_path: {transfer_model}",
                f"  scratch_model_path: {scratch_model}",
                "training:",
                "  image_size: 32",
                "  run_training: false",
                "  local_feasibility_required: false",
                "adoption_gates: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result["summary"].read_text(encoding="utf-8"))
    finetuned = dict(summary["finetuned_evaluation"])
    comparison = dict(summary["finetuned_comparison"])

    assert finetuned["status"] == "completed"
    assert "metric_rows" in finetuned
    assert "expected_metric_rows" in finetuned
    assert "metrics_csv" in finetuned
    assert "masks_dir" in finetuned
    assert "overlays_dir" in finetuned
    assert "proposal_boxes_csv" in finetuned
    assert "proposal_recall_csv" in finetuned
    assert "prompt_failure_count" in finetuned
    assert "prompt_failures_csv" in finetuned
    assert "prompt_failures_json" in finetuned
    assert finetuned["overlay_review_status"] == "passed"
    assert Path(finetuned["metrics_csv"]).exists()
    assert Path(finetuned["prompt_failures_csv"]).exists()
    assert Path(finetuned["proposal_boxes_csv"]).exists()
    assert Path(finetuned["proposal_recall_csv"]).exists()
    assert Path(finetuned["overlays_dir"]).exists()
    assert Path(finetuned["masks_dir"]).exists()
    assert "oracle_dice_gap" in comparison
    assert "adoption_tier" in comparison
    assert "improves_current_auto" in comparison
    assert "improves_current_segmenter" in comparison
    assert "beats_trivial_baseline" in comparison
    assert comparison["improves_current_auto"] is True
    assert comparison["improves_current_segmenter"] is True
    assert comparison["adoption_tier"] == "blocked"
    assert comparison["failure_mode"] == "training_quality"


def test_workflow_summary_fails_closed_when_finetuned_inference_is_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path
    image_path = runtime_root / "raw_data/cohorts/demo/images/img.png"
    mask_path = runtime_root / "raw_data/cohorts/demo/masks/mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    _write_png(mask_path, mask)

    split_row = {
        "manifest_row_id": "row-1",
        "cohort_id": "demo",
        "lane_assignment": "manual_mask_core",
        "source_sample_id": "sample-1",
        "split_group_id": "sample-1",
        "image_path": "raw_data/cohorts/demo/images/img.png",
        "mask_path": "raw_data/cohorts/demo/masks/mask.png",
        "split": "train",
        "selection_rank": 1,
        "selection_reason": "test",
    }
    split_dir = runtime_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame([{**split_row, "split": "validation"}]).to_csv(
        split_dir / "validation.csv", index=False
    )
    pd.DataFrame([{**split_row, "split": "test"}]).to_csv(
        split_dir / "test.csv", index=False
    )

    manifest_path = runtime_root / "raw_data/cohorts/manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / "models/medsam/base.pth"
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b"base")
    medsam_repo = runtime_root / "vendor/MedSAM"
    medsam_repo.mkdir(parents=True, exist_ok=True)
    (medsam_repo / "MedSAM_Inference.py").write_text("# stub\n", encoding="utf-8")
    medsam_python = runtime_root / "bin/python"
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    checkpoint_root = runtime_root / "models/medsam_glomeruli/test-run"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "finetuned_epoch_1.pth").write_bytes(b"finetuned")
    transfer_model = runtime_root / "models/current/transfer.pkl"
    scratch_model = runtime_root / "models/current/scratch.pkl"
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b"transfer")
    scratch_model.write_bytes(b"scratch")

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        "eq.data_management.model_loading.load_model_safely",
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability",
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )

    def _fake_partial_batch(**kwargs):
        failures_path = Path(kwargs["output_dir"]) / "medsam_batch_failures.json"
        failures_path.write_text(
            json.dumps(
                [
                    {
                        "manifest_row_id": "row-1",
                        "image_path": str(image_path),
                        "failure_reason": "simulated prompt failure",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return [
            {
                "manifest_row_id": "row-1",
                "image_path": str(image_path),
                "failure_reason": "simulated prompt failure",
            }
        ]

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch",
        _fake_partial_batch,
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies",
        lambda _cfg: {
            "medsam_python": str(medsam_python),
            "medsam_repo": str(medsam_repo),
            "base_checkpoint": str(base_checkpoint),
            "entrypoint": "",
            "adaptation_mode": "frozen_image_encoder_mask_decoder",
            "training_backend": "eq_native_adapter",
        },
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config",
        lambda **_kwargs: ["python", "-m", "eq.evaluation.medsam_glomeruli_adapter"],
    )

    config_path = runtime_root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "workflow: medsam_glomeruli_fine_tuning",
                "run:",
                "  name: test-run",
                f"  runtime_root_default: {runtime_root}",
                "medsam:",
                f"  python: {medsam_python}",
                f"  repo: {medsam_repo}",
                f"  base_checkpoint: {base_checkpoint}",
                "  device: cpu",
                "  adaptation_mode: frozen_image_encoder_mask_decoder",
                "  frozen_components: [image_encoder, prompt_encoder]",
                "  trainable_components: [mask_decoder]",
                "inputs:",
                f"  manifest_path: {manifest_path.relative_to(runtime_root)}",
                "  lane_assignments: [manual_mask_core]",
                "  split_manifests:",
                f"    train: { (split_dir / 'train.csv').relative_to(runtime_root) }",
                f"    validation: { (split_dir / 'validation.csv').relative_to(runtime_root) }",
                f"    test: { (split_dir / 'test.csv').relative_to(runtime_root) }",
                "outputs:",
                "  evaluation_dir: output/test-eval",
                f"  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}",
                f"  generated_mask_release_root: {(runtime_root / 'derived_data/generated_masks/glomeruli/medsam_finetuned/test-run').relative_to(runtime_root)}",
                f"  generated_mask_registry_path: {(runtime_root / 'derived_data/generated_masks/glomeruli/manifest.csv').relative_to(runtime_root)}",
                "baseline_evaluation:",
                "  external_baselines: false",
                "current_segmenter:",
                f"  transfer_model_path: {transfer_model}",
                f"  scratch_model_path: {scratch_model}",
                "training:",
                "  image_size: 32",
                "  run_training: false",
                "  local_feasibility_required: false",
                "adoption_gates: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result["summary"].read_text(encoding="utf-8"))

    assert summary["finetuned_evaluation"]["status"] == "failed_no_predictions"
    assert summary["finetuned_evaluation"]["prompt_failure_count"] == 1
    assert summary["finetuned_comparison"]["adoption_tier"] == "blocked"
    assert summary["finetuned_comparison"]["failure_mode"] == "failed_no_predictions"


def test_finetuned_comparison_blocks_and_preserves_failed_partial_predictions() -> None:
    comparison = _build_finetuned_comparison_summary(
        finetuned_evaluation={
            "status": "failed_partial_predictions",
            "metrics_csv": "",
            "overlay_review_status": "not_run",
            "prompt_failure_count": 0,
        },
        external_baselines_summary={},
        baseline_outputs={},
        adoption_gates={},
    )

    assert comparison["adoption_tier"] == "blocked"
    assert comparison["primary_generated_mask_transition_status"] == "blocked"
    assert comparison["recommended_generated_mask_source"] == ""
    assert comparison["failure_mode"] == "failed_partial_predictions"


def test_finetuned_evaluation_sets_failed_partial_predictions_when_some_masks_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_a = tmp_path / "raw_data/cohorts/demo/images/img-a.png"
    image_b = tmp_path / "raw_data/cohorts/demo/images/img-b.png"
    mask_a = tmp_path / "raw_data/cohorts/demo/masks/mask-a.png"
    mask_b = tmp_path / "raw_data/cohorts/demo/masks/mask-b.png"
    image_a.parent.mkdir(parents=True, exist_ok=True)
    mask_a.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_a)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_b)
    binary_mask = np.zeros((12, 12), dtype=np.uint8)
    binary_mask[2:10, 2:10] = 1
    _write_png(mask_a, binary_mask)
    _write_png(mask_b, binary_mask)

    transfer_model = tmp_path / "models/current/transfer.pkl"
    scratch_model = tmp_path / "models/current/scratch.pkl"
    transfer_model.parent.mkdir(parents=True, exist_ok=True)
    transfer_model.write_bytes(b"transfer")
    scratch_model.write_bytes(b"scratch")

    checkpoint = tmp_path / "models/medsam_glomeruli/ckpt/final.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"finetuned")

    class _FakeLearner:
        class _Model:
            def eval(self) -> None:
                return None

        model = _Model()

    monkeypatch.setattr(
        "eq.data_management.model_loading.load_model_safely",
        lambda *_args, **_kwargs: _FakeLearner(),
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._predict_probability",
        lambda **_kwargs: (np.ones((12, 12), dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_medsam_inference_paths",
        lambda _cfg: {
            "medsam_python": tmp_path / "bin/python",
            "medsam_repo": tmp_path / "vendor/MedSAM",
            "checkpoint": tmp_path / "models/medsam/base.pth",
            "medsam_script": tmp_path / "vendor/MedSAM/MedSAM_Inference.py",
        },
    )

    def _fake_partial_batch(**kwargs):
        for index, item in enumerate(kwargs["items"]):
            if index > 0:
                break
            Path(item["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ones((12, 12), dtype=np.uint8) * 255).save(
                item["output_path"]
            )
        failures_path = Path(kwargs["output_dir"]) / "medsam_batch_failures.json"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text("[]", encoding="utf-8")
        return []

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._run_medsam_batch",
        _fake_partial_batch,
    )

    fixed_rows = pd.DataFrame(
        [
            {
                "manifest_row_id": "row-1",
                "cohort_id": "demo",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "sample-1",
                "image_path": str(image_a),
                "mask_path": str(mask_a),
                "image_path_resolved": str(image_a),
                "mask_path_resolved": str(mask_a),
                "selection_rank": 1,
                "selection_reason": "test",
            },
            {
                "manifest_row_id": "row-2",
                "cohort_id": "demo",
                "lane_assignment": "manual_mask_core",
                "source_sample_id": "sample-2",
                "image_path": str(image_b),
                "mask_path": str(mask_b),
                "image_path_resolved": str(image_b),
                "mask_path_resolved": str(mask_b),
                "selection_rank": 2,
                "selection_reason": "test",
            },
        ]
    )
    summary = _run_finetuned_fixed_split_evaluation(
        fixed_rows=fixed_rows,
        evaluation_dir=tmp_path / "output/test-eval",
        config={
            "proposal": {"thresholds": [0.35]},
            "tiling": {},
            "current_segmenter": {
                "transfer_model_path": str(transfer_model),
                "scratch_model_path": str(scratch_model),
            },
            "medsam": {"device": "cpu"},
        },
        checkpoint_provenance={
            "supported_checkpoint": True,
            "checkpoint_files": [str(checkpoint)],
        },
        dry_run=False,
    )

    assert summary["status"] == "failed_partial_predictions"
    assert summary["metric_rows"] == 1
    assert summary["expected_metric_rows"] == 2
    assert summary["prompt_failure_count"] == 0


def test_workflow_summary_marks_finetuned_evaluation_skipped_without_supported_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_root = tmp_path
    image_path = runtime_root / "raw_data/cohorts/demo/images/img.png"
    mask_path = runtime_root / "raw_data/cohorts/demo/masks/mask.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(image_path)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 3:9] = 1
    _write_png(mask_path, mask)

    split_row = {
        "manifest_row_id": "row-1",
        "cohort_id": "demo",
        "lane_assignment": "manual_mask_core",
        "source_sample_id": "sample-1",
        "split_group_id": "sample-1",
        "image_path": "raw_data/cohorts/demo/images/img.png",
        "mask_path": "raw_data/cohorts/demo/masks/mask.png",
        "split": "train",
        "selection_rank": 1,
        "selection_reason": "test",
    }
    split_dir = runtime_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame([{**split_row, "split": "validation"}]).to_csv(
        split_dir / "validation.csv", index=False
    )
    pd.DataFrame([{**split_row, "split": "test"}]).to_csv(
        split_dir / "test.csv", index=False
    )

    manifest_path = runtime_root / "raw_data/cohorts/manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([split_row]).to_csv(manifest_path, index=False)

    base_checkpoint = runtime_root / "models/medsam/base.pth"
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.write_bytes(b"base")
    medsam_repo = runtime_root / "vendor/MedSAM"
    medsam_repo.mkdir(parents=True, exist_ok=True)
    medsam_python = runtime_root / "bin/python"
    medsam_python.parent.mkdir(parents=True, exist_ok=True)
    medsam_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    checkpoint_root = runtime_root / "models/medsam_glomeruli/test-run"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._preflight_training_dependencies",
        lambda _cfg: {
            "medsam_python": str(medsam_python),
            "medsam_repo": str(medsam_repo),
            "base_checkpoint": str(base_checkpoint),
            "entrypoint": "",
            "adaptation_mode": "frozen_image_encoder_mask_decoder",
            "training_backend": "eq_native_adapter",
        },
    )
    monkeypatch.setattr(
        "eq.evaluation.run_medsam_glomeruli_fine_tuning_workflow._training_command_for_config",
        lambda **_kwargs: ["python", "-m", "eq.evaluation.medsam_glomeruli_adapter"],
    )

    config_path = runtime_root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "workflow: medsam_glomeruli_fine_tuning",
                "run:",
                "  name: test-run",
                f"  runtime_root_default: {runtime_root}",
                "medsam:",
                f"  python: {medsam_python}",
                f"  repo: {medsam_repo}",
                f"  base_checkpoint: {base_checkpoint}",
                "  device: cpu",
                "  adaptation_mode: frozen_image_encoder_mask_decoder",
                "  frozen_components: [image_encoder, prompt_encoder]",
                "  trainable_components: [mask_decoder]",
                "inputs:",
                f"  manifest_path: {manifest_path.relative_to(runtime_root)}",
                "  lane_assignments: [manual_mask_core]",
                "  split_manifests:",
                f"    train: { (split_dir / 'train.csv').relative_to(runtime_root) }",
                f"    validation: { (split_dir / 'validation.csv').relative_to(runtime_root) }",
                f"    test: { (split_dir / 'test.csv').relative_to(runtime_root) }",
                "outputs:",
                "  evaluation_dir: output/test-eval",
                f"  checkpoint_root: {checkpoint_root.relative_to(runtime_root)}",
                f"  generated_mask_release_root: {(runtime_root / 'derived_data/generated_masks/glomeruli/medsam_finetuned/test-run').relative_to(runtime_root)}",
                f"  generated_mask_registry_path: {(runtime_root / 'derived_data/generated_masks/glomeruli/manifest.csv').relative_to(runtime_root)}",
                "baseline_evaluation:",
                "  external_baselines: false",
                "training:",
                "  image_size: 32",
                "  run_training: false",
                "  local_feasibility_required: false",
                "adoption_gates: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_medsam_glomeruli_fine_tuning_workflow(config_path, dry_run=False)
    summary = json.loads(result["summary"].read_text(encoding="utf-8"))

    assert (
        summary["finetuned_evaluation"]["status"]
        == "skipped_missing_supported_checkpoint"
    )
