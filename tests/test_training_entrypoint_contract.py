from pathlib import Path
import importlib
import json

import cv2
import numpy as np
import pytest

from eq.training import transfer_learning

train_mitochondria = importlib.import_module("eq.training.train_mitochondria")
train_glomeruli = importlib.import_module("eq.training.train_glomeruli")


class _FakeDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)


class _FakeDls:
    def __init__(self, items):
        self.train_ds = _FakeDataset(items[:4])
        self.valid_ds = _FakeDataset(items[4:])
        self.device = "cpu"


class _FakeRecorder:
    values = [[0.5, 0.4, 0.6]]
    losses = [0.5]
    metric_names = ["train_loss", "valid_loss", "dice"]
    log = [0.5, 0.4, 0.6]
    lrs = [1e-3]


class _FakeLearner:
    def __init__(self, dls):
        self.dls = dls
        self.train_ds = dls.train_ds
        self.valid_ds = dls.valid_ds
        self.loss_func = "fake_loss"
        self.recorder = _FakeRecorder()
        self.fit_calls = []
        self.fp16_called = False

    def fit_one_cycle(self, epochs, lr_max, cbs=None):
        self.fit_calls.append((epochs, lr_max, cbs))

    def freeze(self):
        return None

    def unfreeze(self):
        return None

    def to_fp16(self):
        self.fp16_called = True
        return self


def _make_full_image_root(root: Path, count: int = 8) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for i in range(count):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[20:60, 20:60, :] = 180
        cv2.imwrite(str(images / f"sample_{i}.jpg"), image)
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[24:56, 24:56] = 255
        cv2.imwrite(str(masks / f"sample_{i}_mask.png"), mask)


def _patch_artifact_writers(monkeypatch, module):
    monkeypatch.setattr(module, "attach_best_model_callback", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "save_plots", lambda *args, **kwargs: None)

    def fake_export(learn, output_path, model_folder_name):
        model_path = Path(output_path) / f"{model_folder_name}.pkl"
        model_path.write_text("fake model")
        return model_path

    monkeypatch.setattr(module, "export_final_model", fake_export)


def test_mitochondria_training_entrypoint_smoke_uses_dynamic_full_image_root(tmp_path, monkeypatch):
    data_root = tmp_path / "mitochondria_data" / "training"
    testing_root = tmp_path / "mitochondria_data" / "testing"
    _make_full_image_root(data_root)
    _make_full_image_root(testing_root, count=2)
    output_root = tmp_path / "models"
    captured = {}

    def fake_build_dls(root, **kwargs):
        captured["dls_root"] = Path(root)
        captured["dls_kwargs"] = kwargs
        items = sorted((Path(root) / "images").glob("*.jpg"))
        return _FakeDls(items)

    def fake_unet_learner(dls, *args, **kwargs):
        captured["dls"] = dls
        captured["unet_kwargs"] = kwargs
        return _FakeLearner(dls)

    monkeypatch.setattr(train_mitochondria, "build_segmentation_dls_dynamic_patching", fake_build_dls)
    monkeypatch.setattr(train_mitochondria, "unet_learner", fake_unet_learner)
    _patch_artifact_writers(monkeypatch, train_mitochondria)

    learn, model_path = train_mitochondria.train_mitochondria_with_datablock(
        data_dir=str(data_root),
        output_dir=str(output_root),
        model_name="mito_smoke",
        epochs=1,
        batch_size=2,
        image_size=64,
    )

    assert model_path.exists()
    assert learn.fit_calls
    assert captured["dls"] is learn.dls
    assert captured["dls_root"] == data_root
    assert captured["unet_kwargs"]["pretrained"] is True
    metadata_files = list(output_root.glob("**/*_run_metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["training_mode"] == "dynamic_full_image_patching"
    assert metadata["data_root"] == str(data_root)
    assert metadata["model_path"] == str(model_path)
    assert metadata["encoder_initialization"] == "imagenet_pretrained_resnet34"
    assert metadata["candidate_family"] == "mitochondria_no_domain_base"
    assert metadata["training_device"] == "cpu"
    assert metadata["package_versions"]["torch"]
    assert "torchvision" in metadata["package_versions"]
    assert "fastai" in metadata["package_versions"]
    assert "numpy" in metadata["package_versions"]
    assert "command" in metadata
    assert "commit" in metadata["code"]
    assert "dirty" in metadata["code"]

    split_files = list(output_root.glob("**/*_splits.json"))
    assert split_files
    splits = json.loads(split_files[0].read_text())
    split_items = splits["train_images"] + splits["valid_images"]
    assert split_items
    assert all(str(data_root) in item for item in split_items)
    assert all(str(testing_root) not in item for item in split_items)


def test_glomeruli_transfer_training_smoke_validates_full_image_root(tmp_path, monkeypatch):
    data_root = tmp_path / "raw_data" / "glomeruli" / "training_pairs"
    _make_full_image_root(data_root)
    base_model = tmp_path / "mito.pkl"
    base_model.write_text("fake base model")
    (tmp_path / "mito_run_metadata.json").write_text(
        json.dumps(
            {
                "training_mode": "dynamic_full_image_patching",
                "artifact_status": "supported_runtime",
                "data_root": str(tmp_path / "mitochondria_data" / "training"),
                "model_path": str(base_model),
                "command": "unit test",
                "code": {"commit": "test"},
                "package_versions": {"torch": "test"},
                "mitochondria_training_scope": "all_available_pretraining",
                "mitochondria_inference_claim_status": "not_applicable_for_inference_claim",
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "models"
    seen = {}

    def fake_load_model_for_transfer_learning(model_path, target_data_dir, **kwargs):
        seen["model_path"] = Path(model_path)
        seen["target_data_dir"] = Path(target_data_dir)
        items = sorted((Path(target_data_dir) / "images").glob("*.jpg"))
        return _FakeLearner(_FakeDls(items))

    monkeypatch.setattr(
        transfer_learning,
        "load_model_for_transfer_learning",
        fake_load_model_for_transfer_learning,
    )
    _patch_artifact_writers(monkeypatch, transfer_learning)

    learn = transfer_learning.transfer_learn_glomeruli(
        base_model_path=base_model,
        glomeruli_data_dir=data_root,
        output_dir=output_root,
        model_name="glom_smoke",
        epochs=2,
        batch_size=2,
        image_size=64,
        stage1_epochs=1,
        stage2_epochs=1,
        use_lr_find=False,
    )

    assert seen["model_path"] == base_model
    assert seen["target_data_dir"] == data_root
    assert len(learn.fit_calls) == 2
    metadata_files = list(output_root.glob("**/*_run_metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["training_mode"] == "dynamic_full_image_patching"
    assert metadata["data_root"] == str(data_root)
    assert metadata["base_model_path"] == str(base_model)
    assert metadata["training_device"] == "cpu"
    assert metadata["package_versions"]["torch"]
    assert "model_path" in metadata


def test_transfer_learning_refuses_no_base_candidate_when_base_cannot_load(tmp_path, monkeypatch):
    data_root = tmp_path / "raw_data" / "glomeruli" / "training_pairs"
    _make_full_image_root(data_root)
    base_model = tmp_path / "bad_base.pkl"
    base_model.write_text("not a loadable base model")
    captured = {}

    def fake_build_dls(root, **kwargs):
        items = sorted((Path(root) / "images").glob("*.jpg"))
        return _FakeDls(items)

    monkeypatch.setattr(
        transfer_learning,
        "load_supported_segmentation_artifact_metadata",
        lambda path: {"training_mode": "dynamic_full_image_patching"},
    )
    monkeypatch.setattr(transfer_learning, "build_segmentation_dls_dynamic_patching", fake_build_dls)
    def fake_unet_learner(dls, *args, **kwargs):
        captured["unet_kwargs"] = kwargs
        return _FakeLearner(dls)

    monkeypatch.setattr(transfer_learning, "unet_learner", fake_unet_learner)
    monkeypatch.setattr(
        transfer_learning,
        "load_learner",
        lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("fasttransform")),
    )
    monkeypatch.setattr(transfer_learning.torch, "load", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="refusing to continue as a no-base scratch candidate"):
        transfer_learning.load_model_for_transfer_learning(
            base_model,
            data_root,
            batch_size=2,
            image_size=64,
        )

    assert captured["unet_kwargs"]["pretrained"] is False


def test_glomeruli_transfer_entrypoint_refuses_missing_base_before_scratch_training(tmp_path, monkeypatch):
    data_root = tmp_path / "raw_data" / "glomeruli" / "training_pairs"
    _make_full_image_root(data_root)
    output_root = tmp_path / "models"
    missing_base = tmp_path / "missing_mito.pkl"
    scratch_called = False

    def fail_if_scratch_called(*args, **kwargs):
        nonlocal scratch_called
        scratch_called = True
        raise AssertionError("missing transfer base must not fall back to scratch training")

    monkeypatch.setattr(train_glomeruli, "train_glomeruli_with_datablock", fail_if_scratch_called)

    with pytest.raises(FileNotFoundError, match="Refusing to continue as a no-base scratch candidate"):
        train_glomeruli.train_glomeruli_with_transfer_learning(
            data_dir=str(data_root),
            output_dir=str(output_root),
            model_name="glom_transfer",
            base_model_path=str(missing_base),
            epochs=1,
            batch_size=2,
            image_size=64,
        )

    assert scratch_called is False


def test_transfer_learning_refuses_no_base_candidate_when_base_has_no_compatible_weights(tmp_path, monkeypatch):
    data_root = tmp_path / "raw_data" / "glomeruli" / "training_pairs"
    _make_full_image_root(data_root)
    base_model = tmp_path / "incompatible_base.pkl"
    base_model.write_text("fake incompatible base model")

    class _ModelLearner(_FakeLearner):
        def __init__(self, dls, model):
            super().__init__(dls)
            self.model = model

    def fake_build_dls(root, **kwargs):
        items = sorted((Path(root) / "images").glob("*.jpg"))
        return _FakeDls(items)

    monkeypatch.setattr(
        transfer_learning,
        "load_supported_segmentation_artifact_metadata",
        lambda path: {"training_mode": "dynamic_full_image_patching"},
    )
    monkeypatch.setattr(transfer_learning, "build_segmentation_dls_dynamic_patching", fake_build_dls)
    monkeypatch.setattr(
        transfer_learning,
        "unet_learner",
        lambda dls, *args, **kwargs: _ModelLearner(dls, transfer_learning.torch.nn.Linear(2, 2)),
    )
    monkeypatch.setattr(
        transfer_learning,
        "load_learner",
        lambda *args, **kwargs: _ModelLearner(_FakeDls([]), transfer_learning.torch.nn.Linear(3, 3)),
    )
    monkeypatch.setattr(
        transfer_learning.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not a checkpoint")),
    )

    with pytest.raises(RuntimeError, match="copied 0 compatible model parameters"):
        transfer_learning.load_model_for_transfer_learning(
            base_model,
            data_root,
            batch_size=2,
            image_size=64,
            load_encoder_only=False,
        )


def test_transfer_learning_enables_fp16_only_on_cuda(monkeypatch):
    learn = _FakeLearner(_FakeDls([]))

    monkeypatch.setattr(transfer_learning.torch.cuda, "is_available", lambda: False)
    same = transfer_learning._maybe_enable_cuda_fp16(learn)
    assert same is learn
    assert learn.fp16_called is False

    learn_cuda = _FakeLearner(_FakeDls([]))
    monkeypatch.setattr(transfer_learning.torch.cuda, "is_available", lambda: True)
    same_cuda = transfer_learning._maybe_enable_cuda_fp16(learn_cuda)
    assert same_cuda is learn_cuda
    assert learn_cuda.fp16_called is True


def test_glomeruli_scratch_training_preserves_requested_crop_size(tmp_path, monkeypatch):
    data_root = tmp_path / "raw_data" / "glomeruli" / "training_pairs"
    _make_full_image_root(data_root)
    output_root = tmp_path / "models"
    captured = {}

    def fake_batch_size(stage, *, image_size=None, crop_size=None, requested_batch_size=None, mode="production"):
        captured["batch_size_call"] = {
            "stage": stage,
            "image_size": image_size,
            "crop_size": crop_size,
            "requested_batch_size": requested_batch_size,
            "mode": mode,
        }
        return 2

    def fake_build_dls(root, **kwargs):
        captured["dls_call"] = {"root": Path(root), **kwargs}
        items = sorted((Path(root) / "images").glob("*.jpg"))
        return _FakeDls(items)

    def fake_unet_learner(dls, *args, **kwargs):
        captured["dls"] = dls
        captured["unet_kwargs"] = kwargs
        return _FakeLearner(dls)

    monkeypatch.setattr(train_glomeruli, "get_segmentation_training_batch_size", fake_batch_size)
    monkeypatch.setattr(train_glomeruli, "build_segmentation_dls_dynamic_patching", fake_build_dls)
    monkeypatch.setattr(train_glomeruli, "unet_learner", fake_unet_learner)
    _patch_artifact_writers(monkeypatch, train_glomeruli)

    learn = train_glomeruli.train_glomeruli_with_datablock(
        data_dir=str(data_root),
        output_dir=str(output_root),
        model_name="glom_scratch",
        epochs=1,
        batch_size=None,
        image_size=64,
        crop_size=96,
        seed=42,
    )

    assert learn.fit_calls
    assert captured["unet_kwargs"]["pretrained"] is True
    assert captured["batch_size_call"]["crop_size"] == 96
    assert captured["dls_call"]["crop_size"] == 96
    assert captured["dls_call"]["output_size"] == 64
    metadata_files = list(output_root.glob("**/*_run_metadata.json"))
    assert metadata_files
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata["invocation"]["crop_size"] == 96
    assert metadata["invocation"]["output_size"] == 64
    assert metadata["encoder_initialization"] == "imagenet_pretrained_resnet34"
    assert metadata["candidate_family"] == "no_mitochondria_base"
    assert metadata["training_device"] == "cpu"


def _make_static_patch_root(root: Path) -> None:
    (root / "image_patches").mkdir(parents=True)
    (root / "mask_patches").mkdir(parents=True)


def test_training_entrypoints_reject_static_patch_roots_before_model_construction(tmp_path, monkeypatch):
    static_root = tmp_path / "static"
    _make_static_patch_root(static_root)
    base_model = tmp_path / "base.pkl"
    base_model.write_text("fake base")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model construction should not be reached")

    monkeypatch.setattr(train_mitochondria, "unet_learner", fail_if_called)
    monkeypatch.setattr(train_glomeruli, "unet_learner", fail_if_called)
    monkeypatch.setattr(transfer_learning, "load_model_for_transfer_learning", fail_if_called)

    with pytest.raises(ValueError, match="Unsupported static patch training root"):
        train_mitochondria.train_mitochondria_with_datablock(
            data_dir=str(static_root),
            output_dir=str(tmp_path / "mito_models"),
            model_name="mito",
        )

    with pytest.raises(ValueError, match="Unsupported static patch training root"):
        train_glomeruli.train_glomeruli_with_datablock(
            data_dir=str(static_root),
            output_dir=str(tmp_path / "glom_models"),
            model_name="glom",
        )

    with pytest.raises(ValueError, match="Unsupported static patch training root"):
        transfer_learning.transfer_learn_glomeruli(
            base_model_path=base_model,
            glomeruli_data_dir=static_root,
            output_dir=tmp_path / "transfer_models",
        )
