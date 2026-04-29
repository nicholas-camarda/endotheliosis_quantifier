from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eq.quantification.input_contract import (
    QuantificationInputContractError,
    grouping_identity_from_scored_table,
    resolve_quantification_input_contract,
    validate_committed_label_override_path,
)
from eq.quantification.run_endotheliosis_quantification_workflow import (
    resolve_endotheliosis_quantification_contract,
)


def _runtime_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "raw_data" / "cohorts"
    data_dir.mkdir(parents=True)
    model = tmp_path / "models" / "glom.pkl"
    model.parent.mkdir(parents=True)
    model.write_text("model", encoding="utf-8")
    output_dir = tmp_path / "output"
    return data_dir, model, output_dir


def test_resolver_rejects_output_tree_label_override_path(tmp_path: Path):
    data_dir, model, output_dir = _runtime_inputs(tmp_path)
    override = (
        tmp_path
        / "output"
        / "quantification_results"
        / "old"
        / "rubric_label_overrides.csv"
    )
    override.parent.mkdir(parents=True)
    override.write_text("subject_image_id,rubric_score\ncase_1,1\n", encoding="utf-8")

    with pytest.raises(QuantificationInputContractError, match="generated quantification outputs"):
        resolve_quantification_input_contract(
            data_dir=data_dir,
            segmentation_model=model,
            output_dir=output_dir,
            label_overrides_path=override,
        )


def test_committed_config_label_override_path_must_use_stable_input_root():
    validate_committed_label_override_path(
        "derived_data/quantification_inputs/reviewed_label_overrides/"
        "endotheliosis_grade_model/rubric_label_overrides.csv"
    )

    with pytest.raises(QuantificationInputContractError, match="stable derived input"):
        validate_committed_label_override_path(
            "output/quantification_results/old/rubric_label_overrides.csv"
        )


def test_yaml_and_direct_resolution_produce_same_contract(tmp_path: Path):
    data_dir, model, output_dir = _runtime_inputs(tmp_path)
    override = (
        tmp_path
        / "derived_data"
        / "quantification_inputs"
        / "reviewed_label_overrides"
        / "endotheliosis_grade_model"
        / "rubric_label_overrides.csv"
    )
    override.parent.mkdir(parents=True)
    override.write_text("subject_image_id,rubric_score\ncase_1,1\n", encoding="utf-8")

    direct = resolve_quantification_input_contract(
        data_dir=data_dir,
        segmentation_model=model,
        output_dir=output_dir,
        score_source="auto",
        label_overrides_path=override,
    )
    yaml_resolved = resolve_endotheliosis_quantification_contract(
        data_dir=data_dir,
        segmentation_model=model,
        output_dir=output_dir,
        score_source="auto",
        label_overrides_path=override,
    )

    assert direct.reference() == yaml_resolved.reference()


def test_grouping_identity_requires_unique_subject_image_ids():
    scored = pd.DataFrame(
        {
            "subject_image_id": ["case_1", "case_1"],
            "subject_id": ["subject_1", "subject_1"],
        }
    )

    with pytest.raises(QuantificationInputContractError, match="duplicate"):
        grouping_identity_from_scored_table(scored)


def test_direct_cli_forwards_label_overrides(monkeypatch, tmp_path: Path):
    import eq.__main__ as main

    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {"done": tmp_path / "done.txt"}

    monkeypatch.setattr(main, "_load_contract_first_quantification", lambda: fake_runner)
    monkeypatch.setattr(main, "_validate_mode_for_command", lambda *args, **kwargs: None)

    args = SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        segmentation_model=str(tmp_path / "model.pkl"),
        output_dir=str(tmp_path / "out"),
        mapping_file=None,
        annotation_source=None,
        score_source="auto",
        label_overrides=str(tmp_path / "overrides.csv"),
        apply_migration=False,
        stop_after="contract",
    )

    main.quant_endo_command(args)

    assert captured["label_overrides_path"] == tmp_path / "overrides.csv"
