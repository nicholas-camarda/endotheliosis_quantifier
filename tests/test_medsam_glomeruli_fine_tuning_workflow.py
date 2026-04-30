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
    _classify_generated_mask_adoption,
    _package_generated_mask_release,
    _preflight_training_dependencies,
    _prepare_medsam_npy_data,
    _run_external_fixed_split_baselines,
    _training_command_for_config,
    _update_generated_mask_registry,
    _validate_split_manifest,
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
