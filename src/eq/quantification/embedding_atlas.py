"""Label-free ROI embedding atlas runner.

This module owns the atlas-specific review renderer because the existing
quantification review helpers are tied to supervised feature/model evidence.
"""

from __future__ import annotations

import html
import importlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from eq.quantification.modeling_contracts import save_json, to_finite_numeric_matrix

LOGGER = logging.getLogger(__name__)

ATLAS_SUBTREE = Path('burden_model') / 'embedding_atlas'
CLAIM_BOUNDARY = (
    'This atlas supports descriptive morphology clustering and review '
    'prioritization only. It does not provide calibrated severity '
    'probabilities, external validity, causal mechanism evidence, or automatic '
    'replacement of human review.'
)
REQUIRED_IDENTITY_COLUMNS = ['subject_id', 'subject_image_id', 'glomerulus_id']
REQUIRED_ROI_PROVENANCE_COLUMNS = [
    'roi_image_path',
    'roi_mask_path',
    'roi_status',
    'roi_bbox_x0',
    'roi_bbox_y0',
    'roi_bbox_x1',
    'roi_bbox_y1',
    'roi_area',
    'roi_fill_fraction',
    'roi_component_selection',
    'roi_union_bbox_width',
    'roi_union_bbox_height',
]
REQUIRED_EMBEDDING_METADATA_KEYS = [
    'segmentation_model_path',
    'expected_size',
    'embedding_dim',
    'pooling',
    'representation',
]
LABEL_LIKE_TOKENS = (
    'score',
    'grade',
    'severe',
    'severity',
    'label',
    'override',
    'review',
    'adjudicat',
    'target',
    'prediction',
    'probability',
    'candidate',
    'fold',
)
SOURCE_LIKE_TOKENS = (
    'cohort',
    'source',
    'lane',
    'batch',
    'treatment',
    'path',
    'file',
    'name',
    'id',
    'date',
    'workbook',
)
ROI_QC_COLUMNS = [
    'roi_bbox_x0',
    'roi_bbox_y0',
    'roi_bbox_x1',
    'roi_bbox_y1',
    'roi_area',
    'roi_fill_fraction',
    'roi_mean_intensity',
    'roi_openness_score',
    'roi_component_count',
    'roi_union_bbox_width',
    'roi_union_bbox_height',
    'roi_largest_component_area_fraction',
]


class AtlasFailClosed(RuntimeError):
    """Raised after durable fail-closed artifacts have been written."""


@dataclass(frozen=True)
class AtlasPaths:
    """Canonical atlas output paths under a quantification output root."""

    quantification_root: Path
    root: Path
    summary: Path
    feature_space: Path
    clusters: Path
    stability: Path
    diagnostics: Path
    evidence: Path
    review_queue: Path

    @classmethod
    def from_quantification_root(cls, quantification_root: Path | str) -> 'AtlasPaths':
        quantification_root = Path(quantification_root).expanduser()
        root = quantification_root / ATLAS_SUBTREE
        return cls(
            quantification_root=quantification_root,
            root=root,
            summary=root / 'summary',
            feature_space=root / 'feature_space',
            clusters=root / 'clusters',
            stability=root / 'stability',
            diagnostics=root / 'diagnostics',
            evidence=root / 'evidence',
            review_queue=root / 'review_queue',
        )

    def mkdirs(self) -> None:
        for path in [
            self.summary,
            self.feature_space,
            self.clusters,
            self.stability,
            self.diagnostics,
            self.evidence,
            self.review_queue,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class AtlasInputs:
    """Loaded atlas input tables and provenance payloads."""

    embeddings: pd.DataFrame
    roi_examples: pd.DataFrame
    learned_roi: pd.DataFrame | None
    embedding_metadata: dict[str, Any]
    learned_roi_metadata: dict[str, Any]
    paths: dict[str, Path]


@dataclass
class FeatureSpace:
    """Prepared finite feature space."""

    space_id: str
    matrix: np.ndarray
    row_frame: pd.DataFrame
    source_columns: list[str]
    transformed_feature_count: int
    scaling_policy: str
    pca_policy: str
    diagnostics: dict[str, Any]


def run_label_free_roi_embedding_atlas(
    config_path: Path | str | dict[str, Any], *, dry_run: bool = False
) -> dict[str, Path]:
    """Run the label-free ROI embedding atlas from a YAML config."""
    config = _load_config(config_path)
    quantification_root = _resolve_quantification_output_root(config)
    atlas_paths = AtlasPaths.from_quantification_root(quantification_root)
    changed: dict[str, Path] = {}

    if dry_run:
        return {
            'atlas_root': atlas_paths.root,
            'atlas_verdict': atlas_paths.summary / 'atlas_verdict.json',
            'atlas_summary': atlas_paths.summary / 'atlas_summary.md',
            'artifact_manifest': atlas_paths.summary / 'artifact_manifest.json',
            'index': atlas_paths.root / 'INDEX.md',
        }

    atlas_paths.mkdirs()
    try:
        inputs = _load_inputs(config, quantification_root)
        blockers: list[str] = []
        warnings: list[str] = []
        changed['method_availability'] = _write_method_availability(atlas_paths)
        blockers.extend(_validate_input_identity_and_provenance(inputs, atlas_paths))
        if blockers:
            _fail_closed(atlas_paths, changed, blockers=blockers, warnings=warnings)

        audit = _build_label_blinding_audit(inputs, config)
        changed['label_blinding_audit'] = save_json(
            audit, atlas_paths.diagnostics / 'label_blinding_audit.json'
        )
        if audit['status'] == 'failed':
            blockers.extend(audit['blockers'])
            _fail_closed(atlas_paths, changed, blockers=blockers, warnings=warnings)

        feature_spaces, feature_changed = _construct_feature_spaces(
            inputs, atlas_paths, config
        )
        changed.update(feature_changed)
        if not feature_spaces:
            blockers.append('No finite approved feature spaces were estimable.')
            _fail_closed(atlas_paths, changed, blockers=blockers, warnings=warnings)

        method_audit = _audit_method_availability()
        missing_required = [
            method_id
            for method_id, payload in method_audit.items()
            if payload['required'] and not payload['available']
        ]
        if missing_required:
            blockers.append(
                'Missing required clustering methods: ' + ', '.join(missing_required)
            )
            _fail_closed(atlas_paths, changed, blockers=blockers, warnings=warnings)

        assignments, cluster_changed = _run_cluster_methods(
            feature_spaces, atlas_paths, config
        )
        changed.update(cluster_changed)
        if assignments.empty:
            blockers.append('No clustering assignments were produced.')
            _fail_closed(atlas_paths, changed, blockers=blockers, warnings=warnings)

        stability, stability_path = _compute_subject_aware_stability(
            feature_spaces, assignments, atlas_paths, config
        )
        changed['cluster_stability'] = stability_path
        diagnostics, diagnostics_path = _write_posthoc_diagnostics(
            assignments, inputs, stability, atlas_paths, config
        )
        changed['cluster_posthoc_diagnostics'] = diagnostics_path
        evidence_changed = _write_evidence_and_review_queue(
            assignments, inputs, diagnostics, feature_spaces, atlas_paths, config
        )
        changed.update(evidence_changed)

        verdict = _build_verdict(
            status='completed',
            feature_spaces=[space.space_id for space in feature_spaces],
            methods=sorted(assignments['method_id'].dropna().unique().tolist()),
            blockers=[],
            warnings=warnings + diagnostics.get('source_artifact_warnings', []),
            review_queue_count=_count_csv_rows(
                atlas_paths.review_queue / 'atlas_adjudication_queue.csv'
            ),
            selected_atlas_view=_selected_view(assignments),
        )
        changed.update(_write_first_read_artifacts(atlas_paths, verdict, changed))
        return changed
    except AtlasFailClosed:
        return changed
    except Exception as exc:
        LOGGER.exception('LABEL_FREE_ATLAS_STATUS=failed')
        blockers = [f'{type(exc).__name__}: {exc}']
        changed.update(
            _write_first_read_artifacts(
                atlas_paths,
                _build_verdict(
                    status='failed',
                    feature_spaces=[],
                    methods=[],
                    blockers=blockers,
                    warnings=[],
                    review_queue_count=0,
                    selected_atlas_view=None,
                ),
                changed,
            )
        )
        raise AtlasFailClosed(str(exc)) from exc


def _load_config(config_path: Path | str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config_path, dict):
        config = dict(config_path)
        config.setdefault('_config_path', '<in-memory-atlas-config>')
        workflow = config.get('workflow') or config.get('workflow_id')
        if workflow != 'label_free_roi_embedding_atlas':
            raise ValueError(
                'Atlas config must use `workflow: label_free_roi_embedding_atlas`.'
            )
        config['workflow'] = 'label_free_roi_embedding_atlas'
        return config
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f'Config does not exist: {config_path}')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Config must be a YAML mapping: {config_path}')
    if config.get('workflow') != 'label_free_roi_embedding_atlas':
        raise ValueError(
            'Atlas config must use `workflow: label_free_roi_embedding_atlas`.'
        )
    config['_config_path'] = str(config_path)
    return config


def _runtime_root(config: dict[str, Any]) -> Path:
    run_cfg = _mapping(config, 'run', allow_empty=True)
    env_name = str(run_cfg.get('runtime_root_env') or 'EQ_RUNTIME_ROOT')
    runtime_value = os.environ.get(env_name) or run_cfg.get('runtime_root_default')
    if not runtime_value:
        return Path.cwd()
    return Path(str(runtime_value)).expanduser()


def _runtime_path(runtime_root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return runtime_root / path


def _mapping(
    config: dict[str, Any], key: str, *, allow_empty: bool = False
) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None and allow_empty:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    return value


def _resolve_quantification_output_root(config: dict[str, Any]) -> Path:
    runtime_root = _runtime_root(config)
    for key in (
        'quantification_output_root',
        'quantification_output_dir',
        'quantification_dir',
        'output_dir',
        'output_root',
    ):
        value = config.get(key)
        if value not in (None, ''):
            return _runtime_path(runtime_root, value)
    for section_name in ('inputs', 'outputs', 'paths', 'run'):
        section = _mapping(config, section_name, allow_empty=True)
        for key in (
            'quantification_output_root',
            'quantification_output_dir',
            'quantification_dir',
            'output_dir',
            'output_root',
        ):
            value = section.get(key)
            if value not in (None, ''):
                return _runtime_path(runtime_root, value)
    raise ValueError(
        'Atlas config must declare a quantification output root in inputs, outputs, '
        'paths, or run.'
    )


def _atlas_input_path(
    config: dict[str, Any], quantification_root: Path, key: str, default: str
) -> Path:
    inputs = _mapping(config, 'inputs', allow_empty=True)
    aliases = {
        'embedding_table': ('embedding_table', 'roi_embeddings', 'embeddings_path'),
        'roi_examples': ('roi_examples', 'roi_scored_examples', 'roi_examples_path'),
        'learned_roi_features': ('learned_roi_features', 'learned_roi_features_path'),
    }
    value = None
    for alias in aliases.get(key, (key,)):
        value = inputs.get(alias)
        if value not in (None, ''):
            break
    artifacts = _mapping(config, 'input_artifacts', allow_empty=True)
    if value in (None, ''):
        artifact_aliases = {
            'embedding_table': 'roi_embeddings',
            'roi_examples': 'roi_scored_examples',
            'learned_roi_features': 'learned_roi_features',
        }
        value = artifacts.get(artifact_aliases.get(key, key))
    if value in (None, ''):
        value = default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return quantification_root / path


def _load_inputs(config: dict[str, Any], quantification_root: Path) -> AtlasInputs:
    embedding_path = _atlas_input_path(
        config, quantification_root, 'embedding_table', 'embeddings/roi_embeddings.csv'
    )
    roi_path = _atlas_input_path(
        config, quantification_root, 'roi_examples', 'roi_crops/roi_scored_examples.csv'
    )
    learned_path = _atlas_input_path(
        config,
        quantification_root,
        'learned_roi_features',
        'burden_model/learned_roi/feature_sets/learned_roi_features.csv',
    )
    embedding_metadata_path = _atlas_input_path(
        config,
        quantification_root,
        'embedding_metadata',
        'embeddings/embedding_metadata.json',
    )
    learned_metadata_path = _atlas_input_path(
        config,
        quantification_root,
        'learned_roi_feature_metadata',
        'burden_model/learned_roi/feature_sets/learned_roi_feature_metadata.json',
    )
    required = {'embedding_table': embedding_path, 'roi_examples': roi_path}
    missing = [
        f'{name}: {path}' for name, path in required.items() if not path.exists()
    ]
    if missing:
        raise FileNotFoundError('Missing required atlas inputs: ' + '; '.join(missing))

    embeddings = pd.read_csv(embedding_path)
    roi_examples = pd.read_csv(roi_path)
    learned_roi = pd.read_csv(learned_path) if learned_path.exists() else None
    return AtlasInputs(
        embeddings=embeddings.copy(),
        roi_examples=roi_examples.copy(),
        learned_roi=learned_roi.copy() if learned_roi is not None else None,
        embedding_metadata=_read_json_if_exists(embedding_metadata_path),
        learned_roi_metadata=_read_json_if_exists(learned_metadata_path),
        paths={
            'embedding_table': embedding_path,
            'roi_examples': roi_path,
            'learned_roi_features': learned_path,
            'embedding_metadata': embedding_metadata_path,
            'learned_roi_feature_metadata': learned_metadata_path,
        },
    )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else {'payload': payload}


def _validate_input_identity_and_provenance(
    inputs: AtlasInputs, atlas_paths: AtlasPaths
) -> list[str]:
    blockers: list[str] = []
    missing_identity = [
        column
        for column in REQUIRED_IDENTITY_COLUMNS
        if column not in inputs.embeddings
    ]
    if missing_identity:
        blockers.append('missing_identity_columns')
        blockers.append(
            'Missing required identity columns: ' + ', '.join(missing_identity)
        )
    strict_missing_roi = [
        column
        for column in REQUIRED_ROI_PROVENANCE_COLUMNS
        if column not in inputs.embeddings
    ]
    provenance_required = [
        'roi_geometry_contract_version',
        'roi_preprocessing_version',
        'roi_threshold_policy',
        'roi_status',
        'artifact_provenance_id',
        'embedding_model_id',
        'embedding_preprocessing_version',
        'feature_lineage_json',
    ]
    missing_provenance = [
        column for column in provenance_required if column not in inputs.embeddings
    ]
    if missing_provenance:
        blockers.append('stale_or_incomplete_provenance')
        blockers.append(
            'Missing hardened provenance columns: ' + ', '.join(missing_provenance)
        )
    missing_roi = strict_missing_roi if not missing_provenance else []
    missing_metadata = [
        key
        for key in REQUIRED_EMBEDDING_METADATA_KEYS
        if key not in inputs.embedding_metadata
    ]
    column_backed_embedding_provenance = {
        'artifact_provenance_id',
        'embedding_model_id',
        'embedding_preprocessing_version',
    }.issubset(inputs.embeddings.columns)
    if missing_metadata and not column_backed_embedding_provenance:
        blockers.append('stale_or_incomplete_provenance')
        blockers.append(
            'Missing embedding provenance metadata fields: '
            + ', '.join(missing_metadata)
        )
    embedding_columns = _encoder_columns(inputs.embeddings)
    if not embedding_columns:
        blockers.append(
            'No encoder embedding columns with prefix `embedding_` were found.'
        )
    duplicate_keys = inputs.embeddings.duplicated(
        [column for column in REQUIRED_IDENTITY_COLUMNS if column in inputs.embeddings]
    )
    if len(duplicate_keys) and bool(duplicate_keys.any()):
        blockers.append('Embedding table has duplicate atlas identity rows.')
    save_json(
        {
            'status': 'failed' if blockers else 'passed',
            'required_identity_columns': REQUIRED_IDENTITY_COLUMNS,
            'required_roi_provenance_columns': REQUIRED_ROI_PROVENANCE_COLUMNS,
            'required_embedding_metadata_keys': REQUIRED_EMBEDDING_METADATA_KEYS,
            'missing_identity_columns': missing_identity,
            'missing_roi_provenance_columns': missing_roi,
            'missing_embedding_metadata_keys': missing_metadata,
            'input_paths': {key: str(path) for key, path in inputs.paths.items()},
            'row_count': int(len(inputs.embeddings)),
            'subject_count': _nunique(inputs.embeddings, 'subject_id'),
        },
        atlas_paths.diagnostics / 'input_provenance_audit.json',
    )
    return blockers


def _build_label_blinding_audit(
    inputs: AtlasInputs, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    frame = inputs.embeddings
    approved = sorted(set(_encoder_columns(frame) + _roi_qc_columns(frame)))
    learned = (
        _learned_feature_columns(inputs.learned_roi)
        if inputs.learned_roi is not None
        else []
    )
    denied = sorted([column for column in frame.columns if _is_denied_column(column)])
    metadata_only = sorted(
        column
        for column in frame.columns
        if column not in approved and column not in denied and column not in learned
    )
    leaked = sorted(column for column in approved if _is_denied_column(column))
    requested = _requested_feature_allowlist(config or {})
    unapproved_numeric = sorted(
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column in requested and column not in approved and column not in denied
    )
    denied_requested = sorted(column for column in requested if column in denied)
    unapproved_requested = sorted(
        column
        for column in requested
        if column not in approved and column not in learned and column not in denied
    )
    unapproved_lineage = _unapproved_feature_lineage(frame, approved)
    blockers = []
    if leaked:
        blockers.append('Denied columns entered feature space: ' + ', '.join(leaked))
    if denied_requested:
        if any(_has_token(column, LABEL_LIKE_TOKENS) for column in denied_requested):
            blockers.append('label_leakage')
        if any(_has_token(column, SOURCE_LIKE_TOKENS) for column in denied_requested):
            blockers.append('source_leakage')
        blockers.append(
            'Denied configured feature columns: ' + ', '.join(denied_requested)
        )
    if unapproved_requested:
        blockers.append(
            'Configured columns lack approved atlas feature lineage: '
            + ', '.join(unapproved_requested[:50])
        )
    if unapproved_lineage:
        blockers.append('unapproved_feature_lineage')
        blockers.append(
            'Feature lineage is not approved: '
            + ', '.join(sorted(unapproved_lineage)[:50])
        )
    if unapproved_numeric:
        blockers.append(
            'Numeric columns lack approved atlas feature lineage: '
            + ', '.join(unapproved_numeric[:50])
        )
    return {
        'status': 'failed'
        if leaked
        or denied_requested
        or unapproved_requested
        or unapproved_lineage
        or unapproved_numeric
        else 'passed',
        'leakage_detected': bool(leaked or denied_requested),
        'denied_column_entered_feature_matrix': bool(leaked),
        'approved_feature_columns': approved,
        'approved_learned_feature_columns': learned,
        'denied_columns': denied,
        'metadata_only_columns': metadata_only,
        'excluded_label_like_columns': denied,
        'excluded_source_like_columns': [
            column for column in denied if _has_token(column, SOURCE_LIKE_TOKENS)
        ],
        'unapproved_feature_columns': unapproved_numeric,
        'denied_requested_columns': denied_requested,
        'unapproved_requested_columns': unapproved_requested,
        'unapproved_feature_lineage': unapproved_lineage,
        'blockers': blockers,
    }


def _requested_feature_allowlist(config: dict[str, Any]) -> list[str]:
    direct = config.get('feature_allowlist')
    if isinstance(direct, list):
        return [str(column) for column in direct]
    feature_spaces = _mapping(config, 'feature_spaces', allow_empty=True)
    nested = feature_spaces.get('feature_allowlist')
    if isinstance(nested, list):
        return [str(column) for column in nested]
    return []


def _unapproved_feature_lineage(
    frame: pd.DataFrame, approved_columns: Sequence[str]
) -> dict[str, str]:
    lineage: dict[str, str] = {}
    if 'feature_lineage_json' in frame:
        for value in frame['feature_lineage_json'].dropna().astype(str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                lineage.update({str(key): str(item) for key, item in parsed.items()})
    allowed = {'frozen_encoder_embedding', 'roi_qc_measurement', 'learned_roi_feature'}
    return {
        column: lineage[column]
        for column in approved_columns
        if column in lineage and lineage[column] not in allowed
    }


def _encoder_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column.startswith('embedding_') and _is_numeric_series(frame[column])
    )


def _roi_qc_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ROI_QC_COLUMNS
        if column in frame.columns and _is_numeric_series(frame[column])
    ]


def _learned_feature_columns(frame: pd.DataFrame | None) -> list[str]:
    if frame is None:
        return []
    return sorted(
        column
        for column in frame.columns
        if (
            column.startswith('learned_current_glomeruli_encoder_')
            or column.startswith('learned_simple_roi_qc_')
            or column.startswith('learned_roi_')
        )
        and _is_numeric_series(frame[column])
    )


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors='coerce').notna().any()


def _has_token(column: str, tokens: tuple[str, ...]) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in tokens)


def _is_denied_column(column: str) -> bool:
    lowered = column.lower()
    if lowered.startswith(
        ('embedding_', 'learned_current_glomeruli_encoder_', 'learned_simple_roi_qc_')
    ):
        return False
    if column in ROI_QC_COLUMNS:
        return False
    return _has_token(column, LABEL_LIKE_TOKENS) or _has_token(
        column, SOURCE_LIKE_TOKENS
    )


def _construct_feature_spaces(
    inputs: AtlasInputs, atlas_paths: AtlasPaths, config: dict[str, Any]
) -> tuple[list[FeatureSpace], dict[str, Path]]:
    changed: dict[str, Path] = {}
    spaces: list[FeatureSpace] = []
    base = inputs.embeddings.reset_index(drop=True).copy()
    base['atlas_row_id'] = np.arange(len(base), dtype=int)

    encoder_columns = _encoder_columns(base)
    if encoder_columns:
        standardized = _make_standardized_space(
            'encoder_standardized', base, encoder_columns
        )
        spaces.append(standardized)
        feature_spaces_cfg = _mapping(config, 'feature_spaces', allow_empty=True)
        encoder_pca_cfg = feature_spaces_cfg.get('encoder_pca', {})
        if not isinstance(encoder_pca_cfg, dict):
            encoder_pca_cfg = {}
        pca_components = int(
            _mapping(config, 'feature_space', allow_empty=True).get(
                'pca_components',
                encoder_pca_cfg.get(
                    'pca_components',
                    min(20, len(encoder_columns), max(len(base) - 1, 1)),
                ),
            )
        )
        pca_space = _make_pca_space(
            'encoder_pca', standardized, n_components=pca_components
        )
        if pca_space is not None:
            spaces.append(pca_space)

    roi_columns = _roi_qc_columns(base)
    if roi_columns:
        spaces.append(
            _make_standardized_space('roi_qc_standardized', base, roi_columns)
        )

    learned_join, learned_meta = _join_learned_roi_features(base, inputs)
    if learned_join is not None:
        learned_columns = _learned_feature_columns(learned_join)
        if learned_columns:
            space = _make_standardized_space(
                'learned_roi_standardized', learned_join, learned_columns
            )
            space.diagnostics['join_metadata'] = learned_meta
            spaces.append(space)

    manifest = {
        'feature_spaces': [space.diagnostics for space in spaces],
        'package_versions': _package_versions(['numpy', 'pandas', 'sklearn']),
        'claim_boundary': CLAIM_BOUNDARY,
    }
    changed['feature_space_manifest'] = save_json(
        manifest, atlas_paths.feature_space / 'feature_space_manifest.json'
    )
    return spaces, changed


def _make_standardized_space(
    space_id: str, frame: pd.DataFrame, columns: list[str]
) -> FeatureSpace:
    matrix = to_finite_numeric_matrix(frame, columns, finite_bound=1e6)
    raw = frame.loc[:, columns].apply(pd.to_numeric, errors='coerce')
    missing_cells = int(raw.isna().sum().sum())
    nonfinite_cells = int(np.size(matrix) - np.isfinite(matrix).sum())
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    zero_variance = stds <= 0
    near_zero = stds < 1e-8
    scaled = (matrix - means) / np.where(stds > 1e-8, stds, 1.0)
    scaled = np.nan_to_num(np.clip(scaled, -1e6, 1e6), copy=False)
    diagnostics = _feature_space_diagnostics(
        space_id=space_id,
        row_frame=frame,
        source_columns=columns,
        matrix=scaled,
        missing_cells=missing_cells,
        nonfinite_cells=nonfinite_cells,
        zero_variance_count=int(zero_variance.sum()),
        near_zero_variance_count=int(near_zero.sum()),
        scaling_policy='z_score_with_zero_variance_guard',
        pca_policy='not_applied',
        transformed_feature_count=int(scaled.shape[1]),
    )
    return FeatureSpace(
        space_id=space_id,
        matrix=scaled,
        row_frame=frame.copy(),
        source_columns=columns,
        transformed_feature_count=int(scaled.shape[1]),
        scaling_policy='z_score_with_zero_variance_guard',
        pca_policy='not_applied',
        diagnostics=diagnostics,
    )


def _make_pca_space(
    space_id: str, standardized: FeatureSpace, *, n_components: int
) -> FeatureSpace | None:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return None
    max_components = min(
        int(n_components),
        standardized.matrix.shape[0] - 1,
        standardized.matrix.shape[1],
    )
    if max_components < 1:
        return None
    pca = PCA(n_components=max_components, random_state=0, svd_solver='full')
    matrix = pca.fit_transform(standardized.matrix)
    diagnostics = _feature_space_diagnostics(
        space_id=space_id,
        row_frame=standardized.row_frame,
        source_columns=standardized.source_columns,
        matrix=matrix,
        missing_cells=standardized.diagnostics['missing_cells'],
        nonfinite_cells=int(np.size(matrix) - np.isfinite(matrix).sum()),
        zero_variance_count=0,
        near_zero_variance_count=0,
        scaling_policy=standardized.scaling_policy,
        pca_policy=f'pca_components_{max_components}',
        transformed_feature_count=int(matrix.shape[1]),
    )
    diagnostics['pca_explained_variance_ratio'] = [
        float(value) for value in pca.explained_variance_ratio_
    ]
    diagnostics['pca_policy'] = {
        'policy': diagnostics['pca_policy'],
        'component_count': int(max_components),
    }
    return FeatureSpace(
        space_id=space_id,
        matrix=matrix,
        row_frame=standardized.row_frame.copy(),
        source_columns=standardized.source_columns,
        transformed_feature_count=int(matrix.shape[1]),
        scaling_policy=standardized.scaling_policy,
        pca_policy=f'pca_components_{max_components}',
        diagnostics=diagnostics,
    )


def _feature_space_diagnostics(
    *,
    space_id: str,
    row_frame: pd.DataFrame,
    source_columns: list[str],
    matrix: np.ndarray,
    missing_cells: int,
    nonfinite_cells: int,
    zero_variance_count: int,
    near_zero_variance_count: int,
    scaling_policy: str,
    pca_policy: str,
    transformed_feature_count: int,
) -> dict[str, Any]:
    return {
        'feature_space_id': space_id,
        'row_count': int(matrix.shape[0]),
        'subject_count': _nunique(row_frame, 'subject_id'),
        'feature_count': int(len(source_columns)),
        'source_feature_count': int(len(source_columns)),
        'transformed_feature_count': int(transformed_feature_count),
        'missing_value_count': int(missing_cells),
        'missing_cells': int(missing_cells),
        'nonfinite_count': int(nonfinite_cells),
        'nonfinite_cells_after_preprocessing': int(nonfinite_cells),
        'zero_variance_count': int(zero_variance_count),
        'near_zero_variance_count': int(near_zero_variance_count),
        'scaling_policy': scaling_policy,
        'pca_policy': pca_policy,
        'source_columns': source_columns,
        'package_versions': _package_versions(['numpy', 'pandas', 'sklearn']),
    }


def _join_learned_roi_features(
    base: pd.DataFrame, inputs: AtlasInputs
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    learned = inputs.learned_roi
    if learned is None:
        return None, {'status': 'not_present'}
    join_keys = [
        column
        for column in REQUIRED_IDENTITY_COLUMNS
        if column in base.columns and column in learned.columns
    ]
    metadata: dict[str, Any] = {
        'status': 'not_joined',
        'join_keys': join_keys,
        'source_table_path': str(inputs.paths['learned_roi_features']),
        'source_row_count': int(len(learned)),
    }
    if not join_keys:
        metadata['reason'] = 'no_unambiguous_join_keys'
        return None, metadata
    if learned.duplicated(join_keys).any():
        metadata['reason'] = 'duplicate_learned_roi_join_keys'
        return None, metadata
    learned_columns = join_keys + _learned_feature_columns(learned)
    joined = base.merge(
        learned.loc[:, learned_columns], how='left', on=join_keys, validate='one_to_one'
    )
    metadata.update(
        {
            'status': 'joined',
            'joined_row_count': int(len(joined)),
            'feature_count': int(len(_learned_feature_columns(joined))),
            'excluded_rows': int(
                joined[_learned_feature_columns(joined)].isna().all(axis=1).sum()
            )
            if _learned_feature_columns(joined)
            else int(len(joined)),
        }
    )
    return joined, metadata


def _audit_method_availability() -> dict[str, dict[str, Any]]:
    methods = {
        'sklearn_pca': ('sklearn.decomposition', 'required', 'dimension_reduction'),
        'sklearn_kmeans': ('sklearn.cluster', 'required', 'clustering'),
        'sklearn_gaussian_mixture': ('sklearn.mixture', 'required', 'clustering'),
        'sklearn_nearest_neighbors': (
            'sklearn.neighbors',
            'required',
            'review_evidence',
        ),
        'hdbscan': ('hdbscan', 'optional', 'clustering'),
        'umap_learn': ('umap', 'optional', 'visualization'),
        'sklearn_tsne': ('sklearn.manifold', 'optional', 'visualization_only'),
    }
    audit: dict[str, dict[str, Any]] = {}
    for method_id, (module_name, requirement, role) in methods.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', None)
            if version is None and module_name.startswith('sklearn'):
                sklearn = importlib.import_module('sklearn')
                version = getattr(sklearn, '__version__', '')
            audit[method_id] = {
                'available': True,
                'method_id': method_id,
                'module': module_name,
                'version': str(version or ''),
                'required': requirement == 'required',
                'role': role,
                'method_role': 'optional_clustering'
                if method_id == 'hdbscan'
                else role,
                'fit_eligible': role != 'visualization_only',
                'failure_reason': '',
            }
        except Exception as exc:
            audit[method_id] = {
                'available': False,
                'method_id': method_id,
                'module': module_name,
                'version': '',
                'required': requirement == 'required',
                'role': role,
                'method_role': 'optional_clustering'
                if method_id == 'hdbscan'
                else role,
                'fit_eligible': False,
                'failure_reason': f'{type(exc).__name__}: {exc}',
            }
    return audit


def _write_method_availability(atlas_paths: AtlasPaths) -> Path:
    audit = _audit_method_availability()
    for alias, canonical in {
        'kmeans': 'sklearn_kmeans',
        'gaussian_mixture': 'sklearn_gaussian_mixture',
        'nearest_neighbors': 'sklearn_nearest_neighbors',
    }.items():
        payload = audit.get(canonical)
        if payload is not None:
            alias_payload = dict(payload)
            alias_payload['method_id'] = alias
            audit[alias] = alias_payload
    return save_json(
        {'methods': list(audit.values()), 'by_method': audit},
        atlas_paths.diagnostics / 'method_availability.json',
    )


def _run_cluster_methods(
    feature_spaces: list[FeatureSpace], atlas_paths: AtlasPaths, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Path]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    settings = _mapping(config, 'clustering', allow_empty=True)
    method_settings = _mapping(config, 'methods', allow_empty=True)
    cluster_counts = [
        int(value)
        for value in settings.get(
            'cluster_counts', method_settings.get('cluster_count_grid', [2, 3, 4, 5])
        )
        if int(value) >= 2
    ]
    run_settings = _mapping(config, 'run', allow_empty=True)
    random_state = int(settings.get('random_state', run_settings.get('seed', 0)))
    assignment_frames: list[pd.DataFrame] = []
    grid_rows: list[dict[str, Any]] = []
    for space in feature_spaces:
        if len(space.row_frame) < 3:
            continue
        valid_counts = [k for k in cluster_counts if k < len(space.row_frame)]
        if not valid_counts:
            continue
        selected_k = _select_k_by_silhouette(space.matrix, valid_counts, random_state)
        for k in valid_counts:
            grid_rows.append(
                {
                    'feature_space_id': space.space_id,
                    'method_id': 'kmeans',
                    'cluster_count': int(k),
                    'selection_rule': 'max_silhouette_label_free',
                    'selected': bool(k == selected_k),
                }
            )
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(space.matrix)
        distances = kmeans.transform(space.matrix)
        confidence = 1.0 / (1.0 + distances[np.arange(len(labels)), labels])
        assignment_frames.append(
            _assignment_frame(space, 'kmeans', labels, confidence, outlier=False)
        )

        gmm = GaussianMixture(n_components=selected_k, random_state=random_state)
        gmm_labels = gmm.fit_predict(space.matrix)
        probs = gmm.predict_proba(space.matrix)
        assignment_frames.append(
            _assignment_frame(
                space,
                'gaussian_mixture',
                gmm_labels,
                probs[np.arange(len(gmm_labels)), gmm_labels],
                outlier=False,
            )
        )
        grid_rows.append(
            {
                'feature_space_id': space.space_id,
                'method_id': 'gaussian_mixture',
                'cluster_count': int(selected_k),
                'selection_rule': 'reuse_selected_kmeans_k_label_free',
                'selected': True,
            }
        )

    assignments = (
        pd.concat(assignment_frames, ignore_index=True)
        if assignment_frames
        else pd.DataFrame()
    )
    changed: dict[str, Path] = {}
    changed['cluster_assignments'] = atlas_paths.clusters / 'cluster_assignments.csv'
    assignments.to_csv(changed['cluster_assignments'], index=False)
    changed['cluster_parameter_grid'] = (
        atlas_paths.clusters / 'cluster_parameter_grid.csv'
    )
    pd.DataFrame(grid_rows).to_csv(changed['cluster_parameter_grid'], index=False)
    return assignments, changed


def _select_k_by_silhouette(
    matrix: np.ndarray, cluster_counts: list[int], random_state: int
) -> int:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: list[tuple[int, float]] = []
    for k in cluster_counts:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            matrix
        )
        score = (
            float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else -1.0
        )
        scores.append((k, score))
    return max(scores, key=lambda item: (item[1], -item[0]))[0]


def _assignment_frame(
    space: FeatureSpace,
    method_id: str,
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    outlier: bool,
) -> pd.DataFrame:
    identity = [
        column
        for column in [
            'atlas_row_id',
            'roi_row_id',
            'subject_id',
            'subject_image_id',
            'glomerulus_id',
            'roi_image_path',
            'roi_mask_path',
        ]
        if column in space.row_frame.columns
    ]
    frame = space.row_frame.loc[:, identity].copy()
    frame['feature_space_id'] = space.space_id
    frame['method_id'] = method_id
    frame['cluster_id'] = labels.astype(str)
    frame['assignment_confidence'] = confidence.astype(float)
    frame['assignment_distance'] = (
        1.0 / np.maximum(confidence.astype(float), 1e-12)
    ) - 1.0
    frame['outlier_or_noise'] = bool(outlier)
    frame['is_outlier_or_noise'] = bool(outlier)
    return frame


def _compute_subject_aware_stability(
    feature_spaces: list[FeatureSpace],
    assignments: pd.DataFrame,
    atlas_paths: AtlasPaths,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    settings = _mapping(config, 'stability', allow_empty=True)
    method_settings = _mapping(config, 'methods', allow_empty=True)
    if not settings and isinstance(method_settings.get('stability'), dict):
        settings = method_settings['stability']
    resamples = int(settings.get('resamples', settings.get('n_resamples', 20)))
    random_state = int(settings.get('random_state', 0))
    rng = np.random.default_rng(random_state)
    by_space = {space.space_id: space for space in feature_spaces}
    records: list[dict[str, Any]] = []
    for (space_id, method_id), group in assignments.groupby(
        ['feature_space_id', 'method_id'], dropna=False
    ):
        space = by_space.get(str(space_id))
        if space is None or method_id != 'kmeans':
            records.append(
                {
                    'feature_space_id': str(space_id),
                    'method_id': str(method_id),
                    'resampling_unit': 'subject_id',
                    'resamples': 0,
                    'row_count': int(len(group)),
                    'subject_count': _nunique(group, 'subject_id'),
                    'adjusted_rand_mean': None,
                    'non_estimable_reason': 'stability_implemented_for_kmeans_primary_view',
                }
            )
            continue
        labels = group.sort_values('atlas_row_id')['cluster_id'].astype(str).to_numpy()
        subjects = space.row_frame['subject_id'].astype(str).to_numpy()
        unique_subjects = np.unique(subjects)
        cluster_count = len(np.unique(labels))
        if len(unique_subjects) < 3 or cluster_count < 2:
            records.append(
                {
                    'feature_space_id': str(space_id),
                    'method_id': str(method_id),
                    'resampling_unit': 'subject_id',
                    'resamples': 0,
                    'row_count': int(len(group)),
                    'subject_count': int(len(unique_subjects)),
                    'adjusted_rand_mean': None,
                    'non_estimable_reason': 'insufficient_subject_or_cluster_support',
                }
            )
            continue
        ari_values: list[float] = []
        for _ in range(resamples):
            sampled_subjects = rng.choice(
                unique_subjects, size=len(unique_subjects), replace=True
            )
            mask = np.isin(subjects, sampled_subjects)
            if int(mask.sum()) <= cluster_count:
                continue
            km = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
            sample_labels = km.fit_predict(space.matrix[mask])
            ari_values.append(float(adjusted_rand_score(labels[mask], sample_labels)))
        records.append(
            {
                'feature_space_id': str(space_id),
                'method_id': str(method_id),
                'resampling_unit': 'subject_id',
                'resamples': int(len(ari_values)),
                'row_count': int(len(group)),
                'subject_count': int(len(unique_subjects)),
                'adjusted_rand_mean': float(np.mean(ari_values))
                if ari_values
                else None,
                'adjusted_rand_min': float(np.min(ari_values)) if ari_values else None,
                'non_estimable_reason': '' if ari_values else 'no_valid_resamples',
            }
        )
    results = []
    for record in records:
        reason = str(record.get('non_estimable_reason') or '')
        results.append(
            {
                **record,
                'stability_metrics': {
                    'adjusted_rand_mean': record.get('adjusted_rand_mean'),
                    'adjusted_rand_min': record.get('adjusted_rand_min'),
                },
                'non_estimable_reasons': [reason] if reason else [],
            }
        )
    payload = {
        'resampling_unit': 'subject_id',
        'row_count': int(assignments['atlas_row_id'].nunique())
        if 'atlas_row_id' in assignments
        else int(len(assignments)),
        'subject_count': _nunique(assignments, 'subject_id'),
        'records': records,
        'results': results,
        'claim_boundary': CLAIM_BOUNDARY,
    }
    path = save_json(payload, atlas_paths.stability / 'cluster_stability.json')
    return payload, path


def _write_posthoc_diagnostics(
    assignments: pd.DataFrame,
    inputs: AtlasInputs,
    stability: dict[str, Any],
    atlas_paths: AtlasPaths,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    meta_columns = [
        column
        for column in [
            'atlas_row_id',
            'score',
            'cohort_id',
            'lane_assignment',
            'source_batch',
            'source_score_sheet',
            'roi_image_path',
            'roi_mask_path',
            'mask_adequacy_status',
            'roi_area',
            'roi_fill_fraction',
            'roi_component_count',
            'roi_openness_score',
        ]
        if column in inputs.embeddings.columns or column == 'atlas_row_id'
    ]
    metadata = inputs.embeddings.reset_index(drop=True).copy()
    metadata['atlas_row_id'] = np.arange(len(metadata), dtype=int)
    joined = assignments.merge(
        metadata.loc[:, meta_columns], on='atlas_row_id', how='left'
    )
    stability_lookup = {
        (record['feature_space_id'], record['method_id']): record
        for record in stability.get('records', [])
    }
    interpretation = _mapping(config, 'interpretation', allow_empty=True)
    nested_thresholds = interpretation.get('candidate_severity_like_group_requires')
    if not isinstance(nested_thresholds, dict):
        nested_thresholds = {}
    flat_thresholds = _mapping(config, 'interpretation_thresholds', allow_empty=True)
    thresholds = {
        'min_rows': int(
            flat_thresholds.get(
                'min_rows',
                nested_thresholds.get(
                    'min_cluster_rows', interpretation.get('min_rows', 10)
                ),
            )
        ),
        'min_subjects': int(
            flat_thresholds.get(
                'min_subjects',
                nested_thresholds.get(
                    'min_cluster_subjects', interpretation.get('min_subjects', 5)
                ),
            )
        ),
        'min_stability_ari': float(
            flat_thresholds.get(
                'min_stability',
                nested_thresholds.get(
                    'min_stability_ari', interpretation.get('min_stability_ari', 0.35)
                ),
            )
        ),
        'max_source_fraction': float(
            flat_thresholds.get(
                'max_source_fraction',
                nested_thresholds.get(
                    'max_source_fraction',
                    interpretation.get('max_source_fraction', 0.8),
                ),
            )
        ),
        'max_artifact_fraction': float(
            flat_thresholds.get(
                'max_artifact_fraction',
                nested_thresholds.get(
                    'max_artifact_dominance_fraction',
                    interpretation.get('max_artifact_fraction', 0.5),
                ),
            )
        ),
        'min_grade_association_strength': float(
            flat_thresholds.get(
                'min_grade_association_strength',
                interpretation.get('min_grade_range', 1.0),
            )
        ),
        'max_missing_asset_fraction': float(
            flat_thresholds.get(
                'max_missing_asset_fraction',
                nested_thresholds.get(
                    'max_missing_representative_asset_fraction',
                    interpretation.get('max_missing_asset_fraction', 1.0),
                ),
            )
        ),
    }
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for keys, group in joined.groupby(
        ['feature_space_id', 'method_id', 'cluster_id'], dropna=False
    ):
        space_id, method_id, cluster_id = [str(value) for value in keys]
        subject_count = _nunique(group, 'subject_id')
        score_values = (
            pd.to_numeric(group.get('score'), errors='coerce')
            if 'score' in group
            else pd.Series(dtype=float)
        )
        source_fraction = _max_category_fraction(group, 'cohort_id')
        stability_record = stability_lookup.get((space_id, method_id), {})
        ari = stability_record.get('adjusted_rand_mean')
        blockers = []
        label = 'candidate_morphology_group'
        if (
            len(group) < thresholds['min_rows']
            or subject_count < thresholds['min_subjects']
        ):
            blockers.append('min_subjects')
            label = 'insufficient_support'
        if ari is None or float(ari) < thresholds['min_stability_ari']:
            blockers.append('min_stability')
            label = 'unstable_group'
        if (
            source_fraction is not None
            and source_fraction > thresholds['max_source_fraction']
        ):
            blockers.append('max_source_fraction')
            label = 'source_sensitive_group'
        artifact_fraction = _artifact_fraction(group)
        if artifact_fraction > thresholds['max_artifact_fraction']:
            blockers.append('max_artifact_fraction')
            label = 'artifact_or_quality_group'
        grade_range = (
            float(score_values.max() - score_values.min())
            if len(score_values.dropna())
            else 0.0
        )
        grade_association_strength = min(1.0, grade_range / 3.0)
        if grade_association_strength < thresholds['min_grade_association_strength']:
            blockers.append('min_grade_association_strength')
        missing_asset_fraction = _missing_asset_fraction(group)
        if (
            missing_asset_fraction > thresholds['max_missing_asset_fraction']
            or thresholds['max_missing_asset_fraction'] <= 0.0
        ):
            blockers.append('max_missing_asset_fraction')
        if (
            not blockers
            and grade_association_strength
            >= thresholds['min_grade_association_strength']
        ):
            label = 'candidate_severity_like_group'
        records.append(
            {
                'feature_space_id': space_id,
                'method_id': method_id,
                'cluster_id': cluster_id,
                'row_count': int(len(group)),
                'subject_count': int(subject_count),
                'score_distribution': _value_counts(group, 'score'),
                'severe_nonsevere_distribution': _severe_distribution(group),
                'cohort_distribution': _value_counts(group, 'cohort_id'),
                'source_batch_distribution': _value_counts(group, 'source_batch'),
                'roi_qc_summary': _numeric_summary(
                    group,
                    [
                        'roi_area',
                        'roi_fill_fraction',
                        'roi_component_count',
                        'roi_openness_score',
                    ],
                ),
                'max_source_fraction': source_fraction,
                'artifact_fraction': float(artifact_fraction),
                'missing_asset_fraction': float(missing_asset_fraction),
                'grade_association_strength': float(grade_association_strength),
                'stability_adjusted_rand_mean': ari,
                'interpretation_label': label,
                'threshold_blockers': blockers,
                'claim_boundary': CLAIM_BOUNDARY,
            }
        )
        if label in {'source_sensitive_group', 'artifact_or_quality_group'}:
            warnings.append(f'{space_id}/{method_id}/{cluster_id}: {label}')
    payload = {
        'thresholds': thresholds,
        'clusters': records,
        'cluster_interpretations': records,
        'original_score_distribution': _value_counts(joined, 'score'),
        'severe_nonsevere_distribution': _severe_distribution(joined),
        'cohort_source_distribution': {
            'cohort_id': _value_counts(joined, 'cohort_id'),
            'source_batch': _value_counts(joined, 'source_batch'),
            'lane_assignment': _value_counts(joined, 'lane_assignment'),
        },
        'roi_qc_summaries': _numeric_summary(
            joined,
            [
                'roi_area',
                'roi_fill_fraction',
                'roi_component_count',
                'roi_openness_score',
            ],
        ),
        'mask_roi_adequacy': _value_counts(joined, 'mask_adequacy_status'),
        'source_artifact_warnings': warnings,
        'claim_boundary': CLAIM_BOUNDARY,
    }
    path = save_json(
        payload, atlas_paths.diagnostics / 'cluster_posthoc_diagnostics.json'
    )
    return payload, path


def _write_evidence_and_review_queue(
    assignments: pd.DataFrame,
    inputs: AtlasInputs,
    diagnostics: dict[str, Any],
    feature_spaces: list[FeatureSpace],
    atlas_paths: AtlasPaths,
    config: dict[str, Any],
) -> dict[str, Path]:
    from sklearn.neighbors import NearestNeighbors

    changed: dict[str, Path] = {}
    metadata = inputs.embeddings.reset_index(drop=True).copy()
    metadata['atlas_row_id'] = np.arange(len(metadata), dtype=int)
    primary = _primary_assignments(assignments)
    representatives: list[pd.DataFrame] = []
    review_rows: list[dict[str, Any]] = []
    by_space = {space.space_id: space for space in feature_spaces}
    for (space_id, method_id, cluster_id), group in primary.groupby(
        ['feature_space_id', 'method_id', 'cluster_id'], dropna=False
    ):
        space = by_space.get(str(space_id))
        if space is None:
            continue
        indices = group['atlas_row_id'].astype(int).to_numpy()
        centroid = space.matrix[indices].mean(axis=0)
        distances = np.linalg.norm(space.matrix[indices] - centroid, axis=1)
        order = np.argsort(distances)
        chosen = indices[order[: min(5, len(order))]]
        rows = metadata[metadata['atlas_row_id'].isin(chosen)].copy()
        rows['feature_space_id'] = space_id
        rows['method_id'] = method_id
        rows['cluster_id'] = cluster_id
        rows['representative_role'] = 'medoid_or_near_medoid'
        rows['distance_to_cluster_centroid'] = [
            float(distances[np.where(indices == idx)[0][0]])
            for idx in rows['atlas_row_id']
        ]
        representatives.append(rows)
        for _, row in rows.iterrows():
            review_rows.append(
                {
                    'priority': 1,
                    'review_priority': 1,
                    'reason_code': 'cluster_representative_review',
                    'atlas_row_id': int(row['atlas_row_id']),
                    'roi_row_id': row.get('roi_row_id', ''),
                    'subject_id': row.get('subject_id', ''),
                    'subject_image_id': row.get('subject_image_id', ''),
                    'glomerulus_id': row.get('glomerulus_id', ''),
                    'feature_space_id': space_id,
                    'method_id': method_id,
                    'cluster_id': cluster_id,
                    'original_score': row.get('score', ''),
                    'nearest_neighbor_evidence': '',
                    'reviewed_anchor_evidence': '',
                    'roi_image_path': row.get('roi_image_path', ''),
                    'roi_mask_path': row.get('roi_mask_path', ''),
                    'roi_path_provenance': 'roi_image_path;roi_mask_path',
                }
            )

    representative_frame = (
        pd.concat(representatives, ignore_index=True)
        if representatives
        else pd.DataFrame()
    )
    representative_path = atlas_paths.evidence / 'cluster_representatives.csv'
    representative_frame.to_csv(representative_path, index=False)
    changed['cluster_representatives'] = representative_path

    nearest_path = atlas_paths.evidence / 'nearest_neighbors.csv'
    nearest = _nearest_neighbor_evidence(primary, metadata, by_space)
    nearest.to_csv(nearest_path, index=False)
    changed['nearest_neighbors'] = nearest_path

    queue = pd.DataFrame(review_rows)
    if not queue.empty and not nearest.empty:
        first_neighbor = nearest.drop_duplicates('atlas_row_id').set_index(
            'atlas_row_id'
        )
        queue['nearest_neighbor_evidence'] = queue['atlas_row_id'].map(
            first_neighbor['neighbor_atlas_row_id'].to_dict()
        )
    queue_path = atlas_paths.review_queue / 'atlas_adjudication_queue.csv'
    queue.to_csv(queue_path, index=False)
    changed['adjudication_queue'] = queue_path

    html_path = _write_html_review(
        atlas_paths, representative_frame, nearest, diagnostics
    )
    changed['embedding_atlas_review'] = html_path
    return changed


def _primary_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return assignments.copy()
    preferred = assignments[
        (assignments['feature_space_id'] == 'encoder_pca')
        & (assignments['method_id'] == 'kmeans')
    ]
    if not preferred.empty:
        return preferred.copy()
    return assignments[
        (assignments['feature_space_id'] == assignments.iloc[0]['feature_space_id'])
        & (assignments['method_id'] == assignments.iloc[0]['method_id'])
    ].copy()


def _nearest_neighbor_evidence(
    assignments: pd.DataFrame, metadata: pd.DataFrame, by_space: dict[str, FeatureSpace]
) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors

    if assignments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (space_id, method_id), group in assignments.groupby(
        ['feature_space_id', 'method_id']
    ):
        space = by_space.get(str(space_id))
        if space is None or len(group) < 2:
            continue
        ids = group['atlas_row_id'].astype(int).to_numpy()
        nn = NearestNeighbors(n_neighbors=min(4, len(ids))).fit(space.matrix[ids])
        distances, indices = nn.kneighbors(space.matrix[ids])
        subjects = (
            metadata.set_index('atlas_row_id')['subject_id'].astype(str).to_dict()
        )
        for row_pos, atlas_row_id in enumerate(ids):
            neighbor_id = ''
            neighbor_distance = ''
            for distance, neighbor_pos in zip(
                distances[row_pos][1:], indices[row_pos][1:]
            ):
                candidate_id = int(ids[neighbor_pos])
                if subjects.get(candidate_id) != subjects.get(int(atlas_row_id)):
                    neighbor_id = candidate_id
                    neighbor_distance = float(distance)
                    break
            rows.append(
                {
                    'atlas_row_id': int(atlas_row_id),
                    'feature_space_id': space_id,
                    'method_id': method_id,
                    'neighbor_atlas_row_id': neighbor_id,
                    'neighbor_distance': neighbor_distance,
                    'same_subject_neighbors_excluded': True,
                }
            )
    return pd.DataFrame(rows)


def _write_html_review(
    atlas_paths: AtlasPaths,
    representatives: pd.DataFrame,
    nearest: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> Path:
    missing_assets = _missing_asset_rows(representatives)
    cluster_rows = diagnostics.get('clusters', [])
    items = []
    for row in cluster_rows:
        items.append(
            '<tr>'
            f'<td>{html.escape(str(row.get("feature_space_id", "")))}</td>'
            f'<td>{html.escape(str(row.get("method_id", "")))}</td>'
            f'<td>{html.escape(str(row.get("cluster_id", "")))}</td>'
            f'<td>{row.get("row_count", "")}</td>'
            f'<td>{row.get("subject_count", "")}</td>'
            f'<td>{html.escape(str(row.get("interpretation_label", "")))}</td>'
            f'<td>{html.escape(", ".join(row.get("threshold_blockers", [])))}</td>'
            '</tr>'
        )
    rep_rows = []
    for _, row in representatives.head(200).iterrows():
        rep_rows.append(
            '<tr>'
            f'<td>{html.escape(str(row.get("atlas_row_id", "")))}</td>'
            f'<td>{html.escape(str(row.get("cluster_id", "")))}</td>'
            f'<td>{html.escape(str(row.get("score", "")))}</td>'
            f'<td>{html.escape(str(row.get("roi_image_path", "")))}</td>'
            f'<td>{html.escape(str(row.get("roi_mask_path", "")))}</td>'
            '</tr>'
        )
    document = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Label-free ROI embedding atlas review</title></head>
<body>
<h1>Label-free ROI embedding atlas review</h1>
<p>{html.escape(CLAIM_BOUNDARY)}</p>
<h2>Cluster summaries</h2>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>Feature space</th><th>Method</th><th>Cluster</th><th>Rows</th><th>Subjects</th><th>Interpretation</th><th>Blocks</th></tr></thead>
<tbody>{''.join(items)}</tbody>
</table>
<h2>Representative ROI rows</h2>
<p>Missing asset rows: {len(missing_assets)}</p>
<table border="1" cellspacing="0" cellpadding="4">
<thead><tr><th>Atlas row</th><th>Cluster</th><th>Original score</th><th>ROI image path</th><th>ROI mask path</th></tr></thead>
<tbody>{''.join(rep_rows)}</tbody>
</table>
</body>
</html>
"""
    path = atlas_paths.evidence / 'embedding_atlas_review.html'
    path.write_text(document, encoding='utf-8')
    save_json(
        {'missing_asset_count': len(missing_assets), 'missing_assets': missing_assets},
        atlas_paths.evidence / 'missing_assets.json',
    )
    return path


def _missing_asset_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for _, row in frame.iterrows():
        missing = []
        for column in ['roi_image_path', 'roi_mask_path']:
            value = row.get(column, '')
            if value and not Path(str(value)).exists():
                missing.append(column)
        if missing:
            rows.append(
                {
                    'atlas_row_id': int(row.get('atlas_row_id', -1)),
                    'missing_fields': missing,
                    'roi_image_path': str(row.get('roi_image_path', '')),
                    'roi_mask_path': str(row.get('roi_mask_path', '')),
                }
            )
    return rows


def _build_verdict(
    *,
    status: str,
    feature_spaces: list[str],
    methods: list[str],
    blockers: list[str],
    warnings: list[str],
    review_queue_count: int,
    selected_atlas_view: dict[str, str] | None,
) -> dict[str, Any]:
    next_action = (
        'Inspect INDEX.md, evidence/embedding_atlas_review.html, and the review queue.'
        if status == 'completed'
        else 'Resolve blockers and rerun the atlas from the YAML config.'
    )
    return {
        'workflow': 'label_free_roi_embedding_atlas',
        'status': status,
        'workflow_status': status,
        'claim_boundary': CLAIM_BOUNDARY,
        'candidate_feature_spaces': feature_spaces,
        'candidate_clustering_methods': methods,
        'selected_atlas_view': selected_atlas_view,
        'blockers': blockers,
        'source_artifact_warnings': warnings,
        'review_queue_count': int(review_queue_count),
        'next_action': next_action,
    }


def _write_first_read_artifacts(
    atlas_paths: AtlasPaths, verdict: dict[str, Any], changed: dict[str, Path]
) -> dict[str, Path]:
    atlas_paths.mkdirs()
    output: dict[str, Path] = {}
    verdict_path = save_json(verdict, atlas_paths.summary / 'atlas_verdict.json')
    output['atlas_verdict'] = verdict_path
    summary_path = atlas_paths.summary / 'atlas_summary.md'
    summary_path.write_text(_render_summary_md(verdict), encoding='utf-8')
    output['atlas_summary'] = summary_path
    manifest_payload = _artifact_manifest(atlas_paths, {**changed, **output})
    manifest_path = save_json(
        manifest_payload, atlas_paths.summary / 'artifact_manifest.json'
    )
    output['artifact_manifest'] = manifest_path
    index_path = atlas_paths.root / 'INDEX.md'
    index_path.write_text(_render_index_md(verdict, manifest_payload), encoding='utf-8')
    output['index'] = index_path
    return output


def _fail_closed(
    atlas_paths: AtlasPaths,
    changed: dict[str, Path],
    *,
    blockers: list[str],
    warnings: list[str],
) -> None:
    changed.update(
        _write_first_read_artifacts(
            atlas_paths,
            _build_verdict(
                status='failed',
                feature_spaces=[],
                methods=[],
                blockers=blockers,
                warnings=warnings,
                review_queue_count=0,
                selected_atlas_view=None,
            ),
            changed,
        )
    )
    raise AtlasFailClosed('; '.join(blockers))


def _artifact_manifest(
    atlas_paths: AtlasPaths, changed: dict[str, Path]
) -> dict[str, Any]:
    artifacts = {}
    for key, path in sorted(changed.items()):
        try:
            relative = Path(path).resolve().relative_to(atlas_paths.root.resolve())
            artifacts[key] = str(relative)
        except ValueError:
            artifacts[key] = str(path)
    return {
        'atlas_root': str(atlas_paths.root),
        'relative_to': str(ATLAS_SUBTREE),
        'artifacts': artifacts,
        'claim_boundary': CLAIM_BOUNDARY,
    }


def _render_summary_md(verdict: dict[str, Any]) -> str:
    blockers = (
        '\n'.join(f'- {item}' for item in verdict.get('blockers', [])) or '- None'
    )
    warnings = (
        '\n'.join(f'- {item}' for item in verdict.get('source_artifact_warnings', []))
        or '- None'
    )
    return f"""# Label-free ROI embedding atlas summary

Status: `{verdict.get('status')}`

{CLAIM_BOUNDARY}

## Selected Atlas View

`{verdict.get('selected_atlas_view') or 'none'}`

## Blockers

{blockers}

## Source / Artifact Warnings

{warnings}

## Next Action

{verdict.get('next_action')}
"""


def _render_index_md(verdict: dict[str, Any], manifest: dict[str, Any]) -> str:
    artifact_lines = '\n'.join(
        f'- `{path}`' for path in manifest.get('artifacts', {}).values()
    )
    return f"""# Label-free ROI embedding atlas

Status: `{verdict.get('status')}`

{CLAIM_BOUNDARY}

## First Read

- `summary/atlas_verdict.json`
- `summary/atlas_summary.md`
- `summary/artifact_manifest.json`
- `evidence/embedding_atlas_review.html`
- `review_queue/atlas_adjudication_queue.csv`

## Artifacts

{artifact_lines}
"""


def _selected_view(assignments: pd.DataFrame) -> dict[str, str] | None:
    primary = _primary_assignments(assignments)
    if primary.empty:
        return None
    return {
        'feature_space_id': str(primary.iloc[0]['feature_space_id']),
        'method_id': str(primary.iloc[0]['method_id']),
    }


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return max(0, int(len(pd.read_csv(path))))
    except Exception:
        return 0


def _nunique(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(frame[column].dropna().astype(str).nunique())


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column]
        .fillna('missing')
        .astype(str)
        .value_counts()
        .items()
    }


def _severe_distribution(frame: pd.DataFrame) -> dict[str, int]:
    if 'score' not in frame:
        return {}
    scores = pd.to_numeric(frame['score'], errors='coerce')
    labels = np.where(scores >= 3, 'severe', 'nonsevere')
    labels = pd.Series(labels, index=frame.index).where(scores.notna(), 'missing')
    return {str(key): int(value) for key, value in labels.value_counts().items()}


def _max_category_fraction(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    counts = frame[column].fillna('missing').astype(str).value_counts()
    if counts.empty:
        return None
    return float(counts.max() / counts.sum())


def _artifact_fraction(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    flags = pd.Series(False, index=frame.index)
    if 'roi_fill_fraction' in frame:
        flags |= (
            pd.to_numeric(frame['roi_fill_fraction'], errors='coerce').fillna(0) < 0.05
        )
    if 'roi_area' in frame:
        area = pd.to_numeric(frame['roi_area'], errors='coerce')
        flags |= area < area.quantile(0.05)
    return float(flags.mean())


def _missing_asset_fraction(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    flags = pd.Series(False, index=frame.index)
    for column in ['roi_image_path', 'roi_mask_path']:
        if column in frame:
            flags |= (
                frame[column]
                .astype(str)
                .map(lambda value: bool(value) and not Path(value).exists())
            )
    return float(flags.mean())


def _numeric_summary(
    frame: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors='coerce').dropna()
        if values.empty:
            continue
        summary[column] = {
            'mean': float(values.mean()),
            'median': float(values.median()),
            'min': float(values.min()),
            'max': float(values.max()),
        }
    return summary


def _package_versions(modules: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, '__version__', ''))
        except Exception as exc:
            versions[module_name] = f'unavailable: {type(exc).__name__}'
    return versions
