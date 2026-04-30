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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from eq.quantification.modeling_contracts import (
    save_json,
    save_supported_sklearn_model,
    to_finite_numeric_matrix,
)

LOGGER = logging.getLogger(__name__)

ATLAS_SUBTREE = Path('burden_model') / 'embedding_atlas'
CLAIM_BOUNDARY = (
    'This atlas supports descriptive morphology clustering and review '
    'prioritization only. It does not provide calibrated severity '
    'probabilities, independent validation evidence, causal mechanism '
    'evidence, or automatic replacement of human review.'
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
BINARY_TRIAGE_SUBTREE = Path('binary_review_triage')
BINARY_TRIAGE_REVIEW_MAX_ROWS = 30
BINARY_TRIAGE_CLAIM_BOUNDARY = (
    'Binary triage outputs prioritize no/low versus moderate/severe review. '
    'They are current-data grouped-development evidence, not independent '
    'validation evidence, calibrated clinical probabilities, causal '
    'explanations, or autonomous endotheliosis grades.'
)


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


@dataclass
class AtlasAdjudicationResult:
    """Validated atlas adjudication evidence and generated artifact summary."""

    status: str
    cluster_review: pd.DataFrame
    flagged_decisions: pd.DataFrame
    score_corrections: pd.DataFrame
    recovered_anchors: pd.DataFrame
    anchor_manifest: pd.DataFrame
    blocked_clusters: pd.DataFrame
    diagnostics: dict[str, Any]
    changed: dict[str, Path]


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
        adjudication = _write_adjudication_and_binary_triage_outputs(
            assignments=assignments,
            inputs=inputs,
            diagnostics=diagnostics,
            feature_spaces=feature_spaces,
            atlas_paths=atlas_paths,
            config=config,
        )
        changed.update(adjudication.changed)

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
            adjudication_summary=adjudication.diagnostics,
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
    gmm_covariance_type = str(
        settings.get('gmm_covariance_type', method_settings.get('gmm_covariance_type', 'diag'))
    )
    gmm_reg_covar = float(
        settings.get('gmm_reg_covar', method_settings.get('gmm_reg_covar', 1e-4))
    )
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
        selected_k, selection_warnings = _select_k_by_silhouette(
            space.matrix, valid_counts, random_state
        )
        for k in valid_counts:
            grid_rows.append(
                {
                    'feature_space_id': space.space_id,
                    'method_id': 'kmeans',
                    'cluster_count': int(k),
                    'selection_rule': 'max_silhouette_label_free',
                    'selected': bool(k == selected_k),
                    'numeric_warning_count': len(selection_warnings.get(k, [])),
                    'numeric_warning_messages': '; '.join(
                        selection_warnings.get(k, [])[:3]
                    ),
                }
            )
        kmeans = KMeans(n_clusters=selected_k, random_state=random_state, n_init=20)
        labels, kmeans_fit_warnings = _capture_runtime_warnings(
            lambda: kmeans.fit_predict(space.matrix)
        )
        distances, kmeans_transform_warnings = _capture_runtime_warnings(
            lambda: kmeans.transform(space.matrix)
        )
        for row in grid_rows:
            if (
                row['feature_space_id'] == space.space_id
                and row['method_id'] == 'kmeans'
                and row['cluster_count'] == int(selected_k)
            ):
                fit_warnings = kmeans_fit_warnings + kmeans_transform_warnings
                row['fit_numeric_warning_count'] = len(fit_warnings)
                row['fit_numeric_warning_messages'] = '; '.join(fit_warnings[:3])
        confidence = 1.0 / (1.0 + distances[np.arange(len(labels)), labels])
        assignment_frames.append(
            _assignment_frame(space, 'kmeans', labels, confidence, outlier=False)
        )

        gmm = GaussianMixture(
            n_components=selected_k,
            random_state=random_state,
            covariance_type=gmm_covariance_type,
            reg_covar=gmm_reg_covar,
        )
        gmm_labels, gmm_fit_warnings = _capture_runtime_warnings(
            lambda: gmm.fit_predict(space.matrix)
        )
        probs, gmm_predict_warnings = _capture_runtime_warnings(
            lambda: gmm.predict_proba(space.matrix)
        )
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
                'covariance_type': gmm_covariance_type,
                'reg_covar': gmm_reg_covar,
                'fit_numeric_warning_count': len(gmm_fit_warnings + gmm_predict_warnings),
                'fit_numeric_warning_messages': '; '.join(
                    (gmm_fit_warnings + gmm_predict_warnings)[:3]
                ),
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
) -> tuple[int, dict[int, list[str]]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: list[tuple[int, float]] = []
    warnings_by_k: dict[int, list[str]] = {}
    for k in cluster_counts:
        labels, warning_messages = _capture_runtime_warnings(
            lambda: KMeans(
                n_clusters=k, random_state=random_state, n_init=10
            ).fit_predict(matrix)
        )
        if len(set(labels)) > 1:
            score, score_warnings = _capture_runtime_warnings(
                lambda: float(silhouette_score(matrix, labels))
            )
        else:
            score = -1.0
            score_warnings = []
        warnings_by_k[int(k)] = warning_messages + score_warnings
        scores.append((k, score))
    return max(scores, key=lambda item: (item[1], -item[0]))[0], warnings_by_k


def _capture_runtime_warnings(operation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        result = operation()
    messages = [
        f'{warning.category.__name__}: {warning.message}' for warning in caught
    ]
    return result, list(dict.fromkeys(messages))


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
        stability_warning_messages: list[str] = []
        for _ in range(resamples):
            sampled_subjects = rng.choice(
                unique_subjects, size=len(unique_subjects), replace=True
            )
            mask = np.isin(subjects, sampled_subjects)
            if int(mask.sum()) <= cluster_count:
                continue
            km = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
            sample_labels, warning_messages = _capture_runtime_warnings(
                lambda: km.fit_predict(space.matrix[mask])
            )
            ari_values.append(float(adjusted_rand_score(labels[mask], sample_labels)))
            if warning_messages:
                stability_warning_messages.extend(warning_messages)
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
                'numeric_warning_count': len(stability_warning_messages),
                'numeric_warning_messages': '; '.join(
                    list(dict.fromkeys(stability_warning_messages))[:3]
                ),
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
                    'roi_usability': '',
                    'morphology_assessment': '',
                    'score_plausibility': '',
                    'case_cluster_fit': '',
                    'review_action': '',
                    'cluster_interpretation': '',
                    'cluster_review_confidence': '',
                    'cluster_notes': '',
                    'reviewer_notes': '',
                    'reviewer_id': '',
                    'reviewed_at': '',
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
    nearest_lookup = (
        nearest.drop_duplicates('atlas_row_id')
        .set_index('atlas_row_id')
        .to_dict('index')
        if not nearest.empty and 'atlas_row_id' in nearest
        else {}
    )
    cases = []
    for _, row in representatives.head(300).iterrows():
        atlas_row_id = int(row.get('atlas_row_id', -1))
        neighbor = nearest_lookup.get(atlas_row_id, {})
        cases.append(
            {
                'review_id': f'atlas-row-{atlas_row_id}',
                'atlas_row_id': atlas_row_id,
                'subject_id': str(row.get('subject_id', '')),
                'subject_image_id': str(row.get('subject_image_id', '')),
                'glomerulus_id': str(row.get('glomerulus_id', '')),
                'feature_space_id': str(row.get('feature_space_id', '')),
                'method_id': str(row.get('method_id', '')),
                'cluster_id': str(row.get('cluster_id', '')),
                'original_score': str(row.get('score', '')),
                'representative_role': str(row.get('representative_role', '')),
                'distance_to_cluster_centroid': row.get(
                    'distance_to_cluster_centroid', ''
                ),
                'nearest_neighbor_atlas_row_id': neighbor.get(
                    'neighbor_atlas_row_id', ''
                ),
                'nearest_neighbor_distance': neighbor.get('neighbor_distance', ''),
                'roi_image_path': str(row.get('roi_image_path', '')),
                'roi_mask_path': str(row.get('roi_mask_path', '')),
                'roi_image_src': _review_image_src(row.get('roi_image_path', '')),
                'roi_mask_src': _review_image_src(row.get('roi_mask_path', '')),
            }
        )
    payload = {
        'claim_boundary': CLAIM_BOUNDARY,
        'clusters': cluster_rows,
        'cases': cases,
        'missing_asset_count': len(missing_assets),
    }
    payload_json = html.escape(json.dumps(payload, allow_nan=False), quote=False)
    static_case_sections = _static_embedding_review_sections(cases)
    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Label-free ROI embedding atlas review</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f6f7f9; }}
header {{ position: sticky; top: 0; z-index: 10; background: #ffffff; border-bottom: 1px solid #d8dde6; padding: 14px 20px; }}
h1 {{ margin: 0 0 6px; font-size: 22px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
.claim {{ margin: 0; max-width: 1100px; color: #52606d; }}
.toolbar {{ display: flex; gap: 8px; align-items: center; margin-top: 12px; flex-wrap: wrap; }}
button {{ border: 1px solid #9fb3c8; background: #ffffff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
button.primary {{ background: #1f5eff; border-color: #1f5eff; color: white; }}
main {{ padding: 18px 20px 40px; }}
table {{ border-collapse: collapse; width: 100%; background: #ffffff; }}
th, td {{ border: 1px solid #d8dde6; padding: 6px 8px; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #eef2f7; }}
.cluster-block {{ margin-top: 22px; }}
.case-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px; }}
.case-card {{ background: #ffffff; border: 1px solid #d8dde6; border-radius: 8px; overflow: hidden; }}
.case-head {{ padding: 10px 12px; background: #eef2f7; display: flex; justify-content: space-between; gap: 10px; }}
.case-meta {{ font-size: 12px; color: #52606d; line-height: 1.45; }}
.image-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px; }}
figure {{ margin: 0; }}
figcaption {{ font-size: 12px; color: #52606d; margin-bottom: 4px; }}
img {{ width: 100%; max-height: 260px; object-fit: contain; background: #111827; border-radius: 4px; }}
.controls {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 10px; border-top: 1px solid #e4e7eb; }}
label {{ display: grid; gap: 3px; font-size: 12px; color: #334e68; }}
select, input, textarea {{ width: 100%; box-sizing: border-box; border: 1px solid #bcccdc; border-radius: 5px; padding: 6px; font: inherit; background: #ffffff; }}
textarea {{ grid-column: 1 / -1; min-height: 58px; resize: vertical; }}
.path {{ grid-column: 1 / -1; font-size: 11px; color: #627d98; overflow-wrap: anywhere; }}
.status {{ color: #52606d; font-size: 13px; }}
</style>
</head>
<body>
<header>
<h1>Label-free ROI embedding atlas review</h1>
<p class="claim">{html.escape(CLAIM_BOUNDARY)}</p>
<div class="toolbar">
<button class="primary" id="downloadCsv" type="button">Export adjudication CSV</button>
<button id="clearSaved" type="button">Clear saved form values</button>
<span class="status" id="saveStatus">Form autosaves in this browser.</span>
</div>
</header>
<main>
<section>
<h2>Cluster summaries</h2>
<table id="clusterTable">
<thead><tr><th>Feature space</th><th>Method</th><th>Cluster</th><th>Rows</th><th>Subjects</th><th>Interpretation</th><th>Blocks</th><th>Max source fraction</th><th>Artifact fraction</th></tr></thead>
<tbody></tbody>
</table>
</section>
<section>
<h2>Adjudication cases</h2>
<p class="status">Missing asset rows: {len(missing_assets)}. Review each ROI/mask pair, set the dropdowns, add notes where useful, then export CSV.</p>
<div id="cases">{static_case_sections}</div>
</section>
</main>
<script id="atlas-data" type="application/json">{payload_json}</script>
<script>
const data = JSON.parse(document.getElementById('atlas-data').textContent);
const storageKey = 'eq.embedding_atlas_review.' + location.pathname;
const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
const caseFields = {{
  roi_usability: ['','usable','bad_crop','bad_mask','tissue_or_image_artifact','unclear'],
  morphology_assessment: ['','mostly_open_lumina','collapsed_or_closed_capillaries','endotheliosis_like_swelling','rbc_heavy','poor_tissue_quality_or_artifact','not_enough_information'],
  score_plausibility: ['','too_low','plausible','too_high','cannot_judge'],
  case_cluster_fit: ['','representative','atypical_but_valid','outlier_or_wrong_cluster','unclear'],
  review_action: ['','accept','flag_score_review','flag_roi_mask_review','exclude_from_anchor','unclear'],
}};
const clusterFields = {{
  cluster_interpretation: ['','real_morphology_cluster','source_or_batch_artifact','roi_or_mask_artifact','mixed_or_uninterpretable'],
  cluster_review_confidence: ['','high','moderate','low'],
}};
function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function optionHtml(value, current) {{
  return `<option value="${{esc(value)}}" ${{value === current ? 'selected' : ''}}>${{esc(value || 'choose')}}</option>`;
}}
function rowState(id) {{
  return saved[id] || {{}};
}}
function persist(id, key, value) {{
  saved[id] = {{...(saved[id] || {{}}), [key]: value}};
  localStorage.setItem(storageKey, JSON.stringify(saved));
  document.getElementById('saveStatus').textContent = 'Saved locally at ' + new Date().toLocaleTimeString();
}}
function renderClusters() {{
  const tbody = document.querySelector('#clusterTable tbody');
  tbody.innerHTML = data.clusters.map(row => `<tr>
    <td>${{esc(row.feature_space_id)}}</td><td>${{esc(row.method_id)}}</td><td>${{esc(row.cluster_id)}}</td>
    <td>${{esc(row.row_count)}}</td><td>${{esc(row.subject_count)}}</td><td>${{esc(row.interpretation_label)}}</td>
    <td>${{esc((row.threshold_blockers || []).join('; '))}}</td><td>${{esc(row.max_source_fraction)}}</td><td>${{esc(row.artifact_fraction)}}</td>
  </tr>`).join('');
}}
function renderCases() {{
  const byCluster = new Map();
  for (const item of data.cases) {{
    const key = `${{item.feature_space_id}} / ${{item.method_id}} / cluster ${{item.cluster_id}}`;
    if (!byCluster.has(key)) byCluster.set(key, []);
    byCluster.get(key).push(item);
  }}
  const root = document.getElementById('cases');
  root.innerHTML = '';
  for (const [cluster, rows] of byCluster.entries()) {{
    const clusterId = 'cluster:' + cluster;
    const clusterState = rowState(clusterId);
    const section = document.createElement('section');
    section.className = 'cluster-block';
    section.innerHTML = `<h2>${{esc(cluster)}}</h2>
      <div class="case-card">
        <div class="controls">
          ${{Object.entries(clusterFields).map(([key, options]) => `<label>${{key.replaceAll('_',' ')}}<select data-review-id="${{esc(clusterId)}}" data-key="${{key}}">${{options.map(value => optionHtml(value, clusterState[key] || '')).join('')}}</select></label>`).join('')}}
          <label>cluster notes<textarea data-review-id="${{esc(clusterId)}}" data-key="cluster_notes">${{esc(clusterState.cluster_notes || '')}}</textarea></label>
        </div>
      </div>
      <div class="case-grid"></div>`;
    const grid = section.querySelector('.case-grid');
    for (const item of rows) {{
      const state = rowState(item.review_id);
      const card = document.createElement('article');
      card.className = 'case-card';
      card.innerHTML = `
        <div class="case-head">
          <strong>Atlas row ${{esc(item.atlas_row_id)}} | score ${{esc(item.original_score)}}</strong>
          <span class="case-meta">subject ${{esc(item.subject_id)}}<br>nearest row ${{esc(item.nearest_neighbor_atlas_row_id)}}</span>
        </div>
        <div class="image-row">
          <figure><figcaption>ROI image</figcaption><img src="${{esc(item.roi_image_src)}}" alt="ROI image for atlas row ${{esc(item.atlas_row_id)}}"></figure>
          <figure><figcaption>ROI mask</figcaption><img src="${{esc(item.roi_mask_src)}}" alt="ROI mask for atlas row ${{esc(item.atlas_row_id)}}"></figure>
        </div>
        <div class="controls">
          ${{Object.entries(caseFields).map(([key, options]) => `<label>${{key.replaceAll('_',' ')}}<select data-review-id="${{esc(item.review_id)}}" data-key="${{key}}">${{options.map(value => optionHtml(value, state[key] || '')).join('')}}</select></label>`).join('')}}
          <label>reviewer id<input data-review-id="${{esc(item.review_id)}}" data-key="reviewer_id" value="${{esc(state.reviewer_id || '')}}"></label>
          <label>review notes<textarea data-review-id="${{esc(item.review_id)}}" data-key="reviewer_notes">${{esc(state.reviewer_notes || '')}}</textarea></label>
          <div class="path">Image: ${{esc(item.roi_image_path)}}<br>Mask: ${{esc(item.roi_mask_path)}}</div>
        </div>`;
      grid.appendChild(card);
    }}
    root.appendChild(section);
  }}
  document.querySelectorAll('select,input,textarea').forEach(element => {{
    element.addEventListener('change', event => persist(event.target.dataset.reviewId, event.target.dataset.key, event.target.value));
    element.addEventListener('input', event => persist(event.target.dataset.reviewId, event.target.dataset.key, event.target.value));
  }});
}}
function exportCsv() {{
  const columns = ['review_id','atlas_row_id','subject_id','subject_image_id','glomerulus_id','feature_space_id','method_id','cluster_id','original_score','nearest_neighbor_atlas_row_id','roi_usability','morphology_assessment','score_plausibility','case_cluster_fit','review_action','cluster_interpretation','cluster_review_confidence','cluster_notes','reviewer_notes','reviewer_id','reviewed_at','roi_image_path','roi_mask_path'];
  const rows = data.cases.map(item => {{
    const state = rowState(item.review_id);
    const clusterState = rowState(`cluster:${{item.feature_space_id}} / ${{item.method_id}} / cluster ${{item.cluster_id}}`);
    return {{...item, ...clusterState, ...state, reviewed_at: state.reviewed_at || new Date().toISOString()}};
  }});
  const csv = [columns.join(',')].concat(rows.map(row => columns.map(col => `"${{String(row[col] ?? '').replaceAll('"','""')}}"`).join(','))).join('\\n');
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'atlas_adjudication_review_export.csv';
  link.click();
  URL.revokeObjectURL(link.href);
}}
document.getElementById('downloadCsv').addEventListener('click', exportCsv);
document.getElementById('clearSaved').addEventListener('click', () => {{
  if (confirm('Clear all locally saved adjudication values for this review page?')) {{
    localStorage.removeItem(storageKey);
    location.reload();
  }}
}});
renderClusters();
renderCases();
</script>
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


def _static_embedding_review_sections(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return '<p>No representative cases were generated for review.</p>'
    case_fields = {
        'roi_usability': ['', 'usable', 'bad_crop', 'bad_mask', 'tissue_or_image_artifact', 'unclear'],
        'morphology_assessment': [
            '',
            'mostly_open_lumina',
            'collapsed_or_closed_capillaries',
            'endotheliosis_like_swelling',
            'rbc_heavy',
            'poor_tissue_quality_or_artifact',
            'not_enough_information',
        ],
        'score_plausibility': ['', 'too_low', 'plausible', 'too_high', 'cannot_judge'],
        'case_cluster_fit': [
            '',
            'representative',
            'atypical_but_valid',
            'outlier_or_wrong_cluster',
            'unclear',
        ],
        'review_action': [
            '',
            'accept',
            'flag_score_review',
            'flag_roi_mask_review',
            'exclude_from_anchor',
            'unclear',
        ],
    }
    cluster_fields = {
        'cluster_interpretation': [
            '',
            'real_morphology_cluster',
            'source_or_batch_artifact',
            'roi_or_mask_artifact',
            'mixed_or_uninterpretable',
        ],
        'cluster_review_confidence': ['', 'high', 'moderate', 'low'],
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in cases:
        key = (
            f"{item.get('feature_space_id')} / {item.get('method_id')} / "
            f"cluster {item.get('cluster_id')}"
        )
        grouped.setdefault(key, []).append(item)
    sections: list[str] = []
    for cluster, rows in grouped.items():
        cluster_controls = ''.join(
            _static_review_select(field, options) for field, options in cluster_fields.items()
        )
        cards = ''.join(
            _static_embedding_case_card(row, case_fields) for row in rows
        )
        sections.append(
            f"""
<section class="cluster-block">
  <h2>{html.escape(cluster)}</h2>
  <section class="case-card">
    <div class="controls">
      {cluster_controls}
      <label>cluster notes<textarea name="cluster_notes"></textarea></label>
    </div>
  </section>
  <div class="case-grid">{cards}</div>
</section>"""
        )
    return ''.join(sections)


def _static_embedding_case_card(
    row: dict[str, Any], case_fields: dict[str, list[str]]
) -> str:
    controls = ''.join(
        _static_review_select(field, options) for field, options in case_fields.items()
    )
    return f"""
<article class="case-card">
  <div class="case-head">
    <strong>Atlas row {html.escape(str(row.get('atlas_row_id', '')))} | score {html.escape(str(row.get('original_score', '')))}</strong>
    <span class="case-meta">subject {html.escape(str(row.get('subject_id', '')))}<br>nearest row {html.escape(str(row.get('nearest_neighbor_atlas_row_id', '')))}</span>
  </div>
  <div class="image-row">
    <figure><figcaption>ROI image</figcaption><img src="{html.escape(str(row.get('roi_image_src', '')))}" alt="ROI image for atlas row {html.escape(str(row.get('atlas_row_id', '')))}"></figure>
    <figure><figcaption>ROI mask</figcaption><img src="{html.escape(str(row.get('roi_mask_src', '')))}" alt="ROI mask for atlas row {html.escape(str(row.get('atlas_row_id', '')))}"></figure>
  </div>
  <div class="controls">
    {controls}
    <label>reviewer id<input name="reviewer_id"></label>
    <label>review notes<textarea name="reviewer_notes"></textarea></label>
    <div class="path">Image: {html.escape(str(row.get('roi_image_path', '')))}<br>Mask: {html.escape(str(row.get('roi_mask_path', '')))}</div>
  </div>
</article>"""


def _static_review_select(field: str, options: list[str]) -> str:
    choices = ''.join(
        f'<option value="{html.escape(value)}">{html.escape(value or "choose")}</option>'
        for value in options
    )
    return (
        f'<label>{html.escape(field.replace("_", " "))}'
        f'<select name="{html.escape(field)}">{choices}</select></label>'
    )


CLUSTER_REVIEW_REQUIRED_COLUMNS = [
    'review_id',
    'atlas_row_id',
    'subject_id',
    'subject_image_id',
    'glomerulus_id',
    'feature_space_id',
    'method_id',
    'cluster_id',
    'original_score',
    'nearest_neighbor_atlas_row_id',
    'roi_usability',
    'morphology_assessment',
    'score_plausibility',
    'case_cluster_fit',
    'review_action',
    'cluster_interpretation',
    'cluster_review_confidence',
    'cluster_notes',
    'reviewer_notes',
    'reviewer_id',
    'reviewed_at',
    'roi_image_path',
    'roi_mask_path',
]

FLAGGED_DECISION_REQUIRED_COLUMNS = [
    'review_id',
    'atlas_row_id',
    'subject_image_id',
    'cluster_id',
    'original_score',
    'review_action',
    'score_plausibility',
    'case_cluster_fit',
    'score_decision',
    'corrected_score',
    'score_error_reason',
    'anchor_decision',
    'anchor_exclusion_reason',
    'final_notes',
    'reviewed_at',
    'roi_image_path',
    'roi_mask_path',
]


def _write_adjudication_and_binary_triage_outputs(
    *,
    assignments: pd.DataFrame,
    inputs: AtlasInputs,
    diagnostics: dict[str, Any],
    feature_spaces: list[FeatureSpace],
    atlas_paths: AtlasPaths,
    config: dict[str, Any],
) -> AtlasAdjudicationResult:
    metadata = inputs.embeddings.reset_index(drop=True).copy()
    metadata['atlas_row_id'] = np.arange(len(metadata), dtype=int)
    primary = _primary_assignments(assignments)
    reference = _atlas_reference_frame(metadata, primary)
    cluster_review_path = _adjudication_input_path(
        config,
        atlas_paths,
        'cluster_review_export',
        'evidence/atlas_adjudication_review_export.csv',
    )
    flagged_decision_path = _adjudication_input_path(
        config,
        atlas_paths,
        'flagged_case_decisions',
        'evidence/atlas_flagged_case_decisions.csv',
    )
    cluster_review = _read_optional_csv(cluster_review_path)
    flagged_decisions = _read_optional_csv(flagged_decision_path)
    changed: dict[str, Path] = {}
    validation_errors: list[str] = []
    if not cluster_review.empty:
        validation_errors.extend(
            _validate_adjudication_rows(
                cluster_review,
                reference,
                required_columns=CLUSTER_REVIEW_REQUIRED_COLUMNS,
                comparison_columns=[
                    'subject_image_id',
                    'cluster_id',
                    'original_score',
                    'roi_image_path',
                    'roi_mask_path',
                ],
                decision_columns=[
                    'roi_usability',
                    'morphology_assessment',
                    'score_plausibility',
                    'case_cluster_fit',
                    'review_action',
                    'cluster_interpretation',
                    'cluster_review_confidence',
                    'cluster_notes',
                ],
                input_name='cluster_review_export',
            )
        )
    if not flagged_decisions.empty:
        validation_errors.extend(
            _validate_adjudication_rows(
                flagged_decisions,
                reference,
                required_columns=FLAGGED_DECISION_REQUIRED_COLUMNS,
                comparison_columns=[
                    'subject_image_id',
                    'cluster_id',
                    'original_score',
                    'roi_image_path',
                    'roi_mask_path',
                ],
                decision_columns=[
                    'score_decision',
                    'corrected_score',
                    'score_error_reason',
                    'anchor_decision',
                    'anchor_exclusion_reason',
                ],
                input_name='flagged_case_decisions',
            )
        )
    status = (
        'failed_validation'
        if validation_errors
        else 'provided'
        if (not cluster_review.empty or not flagged_decisions.empty)
        else 'not_provided'
    )
    adjudication_diagnostics = {
        'status': status,
        'cluster_review_export': str(cluster_review_path),
        'cluster_review_export_exists': cluster_review_path.exists(),
        'cluster_review_rows': int(len(cluster_review)),
        'flagged_case_decisions': str(flagged_decision_path),
        'flagged_case_decisions_exists': flagged_decision_path.exists(),
        'flagged_decision_rows': int(len(flagged_decisions)),
        'validation_errors': validation_errors,
    }
    diagnostics_path = save_json(
        adjudication_diagnostics,
        atlas_paths.diagnostics / 'adjudication_input_diagnostics.json',
    )
    changed['adjudication_input_diagnostics'] = diagnostics_path
    if validation_errors:
        raise ValueError(
            'Invalid atlas adjudication evidence: ' + '; '.join(validation_errors)
        )

    case_actions = _case_actions_from_cluster_review(cluster_review)
    case_actions_path = atlas_paths.evidence / 'atlas_adjudicated_case_actions.csv'
    case_actions.to_csv(case_actions_path, index=False)
    changed['atlas_adjudicated_case_actions'] = case_actions_path
    focused_review_path = _write_flagged_case_review_html(atlas_paths, case_actions)
    changed['atlas_flagged_case_review'] = focused_review_path

    score_corrections = _score_corrections_from_flagged_decisions(flagged_decisions)
    score_correction_path = atlas_paths.evidence / 'atlas_score_corrections.csv'
    score_corrections.to_csv(score_correction_path, index=False)
    changed['atlas_score_corrections'] = score_correction_path

    recovered_anchors = _recovered_anchors_from_flagged_decisions(flagged_decisions)
    recovered_anchor_path = atlas_paths.evidence / 'atlas_recovered_anchor_examples.csv'
    recovered_anchors.to_csv(recovered_anchor_path, index=False)
    changed['atlas_recovered_anchor_examples'] = recovered_anchor_path

    anchor_manifest, blocked_clusters = _anchor_manifests_from_cluster_review(
        cluster_review
    )
    anchor_manifest_path = atlas_paths.evidence / 'atlas_adjudicated_anchor_manifest.csv'
    blocked_cluster_path = atlas_paths.evidence / 'atlas_blocked_cluster_manifest.csv'
    anchor_manifest.to_csv(anchor_manifest_path, index=False)
    blocked_clusters.to_csv(blocked_cluster_path, index=False)
    changed['atlas_adjudicated_anchor_manifest'] = anchor_manifest_path
    changed['atlas_blocked_cluster_manifest'] = blocked_cluster_path

    final_outcome = _final_adjudication_outcome(
        cluster_review_path=cluster_review_path,
        flagged_decision_path=flagged_decision_path,
        cluster_review=cluster_review,
        flagged_decisions=flagged_decisions,
        score_corrections=score_corrections,
        recovered_anchors=recovered_anchors,
        anchor_manifest=anchor_manifest,
        blocked_clusters=blocked_clusters,
        diagnostics=adjudication_diagnostics,
    )
    final_json_path = save_json(
        final_outcome, atlas_paths.evidence / 'atlas_final_adjudication_outcome.json'
    )
    final_md_path = atlas_paths.evidence / 'atlas_final_adjudication_outcome.md'
    final_md_path.write_text(
        _render_final_adjudication_outcome_md(final_outcome), encoding='utf-8'
    )
    changed['atlas_final_adjudication_outcome'] = final_json_path
    changed['atlas_final_adjudication_outcome_md'] = final_md_path

    triage_changed = _write_binary_review_triage_outputs(
        atlas_paths=atlas_paths,
        inputs=inputs,
        assignments=assignments,
        feature_spaces=feature_spaces,
        anchor_manifest=anchor_manifest,
        blocked_clusters=blocked_clusters,
        recovered_anchors=recovered_anchors,
        score_corrections=score_corrections,
    )
    changed.update(triage_changed)
    adjudication_diagnostics.update(
        {
            'score_correction_count': int(len(score_corrections)),
            'recovered_anchor_count': int(len(recovered_anchors)),
            'candidate_anchor_cluster_count': int(len(anchor_manifest)),
            'blocked_cluster_count': int(len(blocked_clusters)),
            'final_outcome_path': str(final_json_path),
            'binary_triage_verdict_path': str(
                triage_changed.get('binary_triage_verdict', '')
            ),
        }
    )
    save_json(
        adjudication_diagnostics,
        atlas_paths.diagnostics / 'adjudication_input_diagnostics.json',
    )
    return AtlasAdjudicationResult(
        status=status,
        cluster_review=cluster_review,
        flagged_decisions=flagged_decisions,
        score_corrections=score_corrections,
        recovered_anchors=recovered_anchors,
        anchor_manifest=anchor_manifest,
        blocked_clusters=blocked_clusters,
        diagnostics=adjudication_diagnostics,
        changed=changed,
    )


def _adjudication_input_path(
    config: dict[str, Any], atlas_paths: AtlasPaths, key: str, default: str
) -> Path:
    adjudication = _mapping(config, 'adjudication', allow_empty=True)
    review = _mapping(config, 'review', allow_empty=True)
    value = adjudication.get(key, review.get(key, default))
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:1] == ('burden_model',):
        return atlas_paths.quantification_root / path
    return atlas_paths.root / path


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna('')


def _atlas_reference_frame(metadata: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in [
            'atlas_row_id',
            'subject_id',
            'subject_image_id',
            'glomerulus_id',
            'score',
            'roi_image_path',
            'roi_mask_path',
            'cohort_id',
        ]
        if column in metadata.columns
    ]
    reference = metadata.loc[:, keep].copy()
    reference['original_score'] = reference.get('score', '')
    primary_keep = [
        column
        for column in [
            'atlas_row_id',
            'feature_space_id',
            'method_id',
            'cluster_id',
            'assignment_confidence',
            'assignment_distance',
        ]
        if column in primary.columns
    ]
    if primary_keep:
        reference = reference.merge(
            primary.loc[:, primary_keep].drop_duplicates('atlas_row_id'),
            on='atlas_row_id',
            how='left',
            suffixes=('', '_assignment'),
        )
    return reference


def _validate_adjudication_rows(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    required_columns: list[str],
    comparison_columns: list[str],
    decision_columns: list[str],
    input_name: str,
) -> list[str]:
    errors: list[str] = []
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        errors.append(f'{input_name}: missing columns {missing}')
        return errors
    work = frame.copy()
    work['atlas_row_id'] = pd.to_numeric(work['atlas_row_id'], errors='coerce')
    if work['atlas_row_id'].isna().any():
        errors.append(f'{input_name}: nonnumeric atlas_row_id values')
        return errors
    work['atlas_row_id'] = work['atlas_row_id'].astype(int)
    ref = reference.drop_duplicates('atlas_row_id').set_index('atlas_row_id')
    missing_ids = sorted(set(work['atlas_row_id']) - set(ref.index))
    if missing_ids:
        errors.append(f'{input_name}: unmatched atlas_row_id values {missing_ids[:20]}')
    duplicate = work[work.duplicated('atlas_row_id', keep=False)]
    if not duplicate.empty:
        for atlas_row_id, group in duplicate.groupby('atlas_row_id'):
            for column in decision_columns:
                if column in group.columns and group[column].astype(str).nunique() > 1:
                    errors.append(
                        f'{input_name}: conflicting duplicate decisions for '
                        f'atlas_row_id {atlas_row_id} column {column}'
                    )
    for _, row in work.iterrows():
        atlas_row_id = int(row['atlas_row_id'])
        if atlas_row_id not in ref.index:
            continue
        ref_row = ref.loc[atlas_row_id]
        for column in comparison_columns:
            if column not in work.columns or column not in ref_row.index:
                continue
            provided = str(row.get(column, '')).strip()
            if provided == '':
                continue
            expected = ref_row.get(column, '')
            if column == 'original_score':
                expected = ref_row.get('score', expected)
                if _score_text(provided) != _score_text(expected):
                    errors.append(
                        f'{input_name}: atlas_row_id {atlas_row_id} original_score '
                        f'{provided} != current {expected}'
                    )
            elif str(provided) != str(expected):
                errors.append(
                    f'{input_name}: atlas_row_id {atlas_row_id} {column} '
                    f'{provided} != current {expected}'
                )
    return errors


def _score_text(value: Any) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    if pd.isna(numeric):
        return str(value)
    return f'{float(numeric):.6g}'


def _case_actions_from_cluster_review(cluster_review: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'atlas_row_id',
        'subject_id',
        'subject_image_id',
        'cluster_id',
        'original_score',
        'morphology_assessment',
        'score_plausibility',
        'case_cluster_fit',
        'review_action',
        'reviewer_notes',
        'roi_image_path',
        'roi_mask_path',
    ]
    if cluster_review.empty:
        return pd.DataFrame(columns=columns)
    actions = cluster_review[
        cluster_review.get('review_action', pd.Series('', index=cluster_review.index))
        .astype(str)
        .isin(['flag_score_review', 'flag_roi_mask_review', 'exclude_from_anchor'])
    ].copy()
    for column in columns:
        if column not in actions.columns:
            actions[column] = ''
    return actions.loc[:, columns].sort_values(['cluster_id', 'atlas_row_id'])


def _score_corrections_from_flagged_decisions(flagged: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'atlas_row_id',
        'subject_image_id',
        'cluster_id',
        'original_score',
        'adjudicated_score',
        'score_decision',
        'reason',
        'final_notes',
        'reviewed_at',
        'roi_image_path',
        'roi_mask_path',
    ]
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    rows = flagged[flagged.get('score_decision', '').astype(str).eq('change_score')].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            'atlas_row_id': rows['atlas_row_id'],
            'subject_image_id': rows['subject_image_id'],
            'cluster_id': rows['cluster_id'],
            'original_score': rows['original_score'],
            'adjudicated_score': rows.get('corrected_score', ''),
            'score_decision': rows.get('score_decision', ''),
            'reason': rows.get('score_error_reason', ''),
            'final_notes': rows.get('final_notes', ''),
            'reviewed_at': rows.get('reviewed_at', ''),
            'roi_image_path': rows.get('roi_image_path', ''),
            'roi_mask_path': rows.get('roi_mask_path', ''),
        }
    )
    return result.loc[:, columns]


def _recovered_anchors_from_flagged_decisions(flagged: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'atlas_row_id',
        'subject_image_id',
        'cluster_id',
        'score',
        'anchor_decision',
        'reason',
        'final_notes',
        'reviewed_at',
        'roi_image_path',
        'roi_mask_path',
    ]
    if flagged.empty or 'anchor_decision' not in flagged.columns:
        return pd.DataFrame(columns=columns)
    rows = flagged[flagged['anchor_decision'].astype(str).str.startswith('allow_as')].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(
        {
            'atlas_row_id': rows['atlas_row_id'],
            'subject_image_id': rows['subject_image_id'],
            'cluster_id': rows['cluster_id'],
            'score': rows['original_score'],
            'anchor_decision': rows['anchor_decision'],
            'reason': rows.get('anchor_exclusion_reason', ''),
            'final_notes': rows.get('final_notes', ''),
            'reviewed_at': rows.get('reviewed_at', ''),
            'roi_image_path': rows.get('roi_image_path', ''),
            'roi_mask_path': rows.get('roi_mask_path', ''),
        }
    )
    return result.loc[:, columns]


def _anchor_manifests_from_cluster_review(
    cluster_review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        'feature_space_id',
        'method_id',
        'cluster_id',
        'adjudication_verdict',
        'dominant_morphology',
        'anchor_class',
        'rows_reviewed',
        'accepted_count',
        'representative_count',
        'score_review_count',
        'exclude_from_anchor_count',
        'cluster_interpretation',
        'cluster_review_confidence',
        'cluster_notes',
        'atlas_row_ids',
        'claim_boundary',
    ]
    if cluster_review.empty:
        empty = pd.DataFrame(columns=columns)
        return empty.copy(), empty.copy()
    records: list[dict[str, Any]] = []
    for (space_id, method_id, cluster_id), group in cluster_review.groupby(
        ['feature_space_id', 'method_id', 'cluster_id'], dropna=False
    ):
        interpretation = _mode_text(group, 'cluster_interpretation')
        notes = _mode_text(group, 'cluster_notes')
        morphology = _mode_text(group, 'morphology_assessment')
        accepted_count = int(group.get('review_action', '').astype(str).eq('accept').sum())
        representative_count = int(
            group.get('case_cluster_fit', '').astype(str).eq('representative').sum()
        )
        score_review_count = int(
            group.get('review_action', '').astype(str).eq('flag_score_review').sum()
        )
        exclude_count = int(
            group.get('review_action', '').astype(str).eq('exclude_from_anchor').sum()
        )
        blocked = interpretation != 'real_morphology_cluster' or exclude_count > 0
        anchor_class = _anchor_class_from_review_text(notes, morphology)
        verdict = (
            'blocked_cluster_anchor'
            if blocked
            else 'candidate_anchor_cluster'
            if anchor_class
            else 'needs_more_review'
        )
        records.append(
            {
                'feature_space_id': str(space_id),
                'method_id': str(method_id),
                'cluster_id': str(cluster_id),
                'adjudication_verdict': verdict,
                'dominant_morphology': morphology,
                'anchor_class': anchor_class,
                'rows_reviewed': int(len(group)),
                'accepted_count': accepted_count,
                'representative_count': representative_count,
                'score_review_count': score_review_count,
                'exclude_from_anchor_count': exclude_count,
                'cluster_interpretation': interpretation,
                'cluster_review_confidence': _mode_text(group, 'cluster_review_confidence'),
                'cluster_notes': notes,
                'atlas_row_ids': ';'.join(group['atlas_row_id'].astype(str).tolist()),
                'claim_boundary': CLAIM_BOUNDARY,
            }
        )
    frame = pd.DataFrame(records, columns=columns)
    anchors = frame[frame['adjudication_verdict'].eq('candidate_anchor_cluster')].copy()
    blocked = frame[frame['adjudication_verdict'].eq('blocked_cluster_anchor')].copy()
    return anchors, blocked


def _mode_text(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return ''
    values = frame[column].fillna('').astype(str)
    values = values[values.str.strip() != '']
    if values.empty:
        return ''
    return str(values.mode().iat[0]).strip()


def _anchor_class_from_review_text(notes: str, morphology: str) -> str:
    text = f'{notes} {morphology}'.lower()
    if 'moderate' in text or 'severe' in text or 'collapsed' in text or 'closed' in text:
        return 'moderate_severe'
    if 'low' in text or 'no endotheliosis' in text or 'open' in text:
        return 'no_low'
    return ''


def _final_adjudication_outcome(
    *,
    cluster_review_path: Path,
    flagged_decision_path: Path,
    cluster_review: pd.DataFrame,
    flagged_decisions: pd.DataFrame,
    score_corrections: pd.DataFrame,
    recovered_anchors: pd.DataFrame,
    anchor_manifest: pd.DataFrame,
    blocked_clusters: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        'source_cluster_adjudication_export': str(cluster_review_path),
        'source_flagged_decisions': str(flagged_decision_path),
        'adjudication_status': diagnostics.get('status', 'not_provided'),
        'reviewed_cluster_rows': int(len(cluster_review)),
        'reviewed_flagged_cases': int(len(flagged_decisions)),
        'score_change_count': int(len(score_corrections)),
        'score_changes': score_corrections.to_dict(orient='records'),
        'score_kept_count': int(
            flagged_decisions.get('score_decision', pd.Series(dtype=str))
            .astype(str)
            .eq('keep_original_score')
            .sum()
        )
        if not flagged_decisions.empty
        else 0,
        'recovered_anchor_count': int(len(recovered_anchors)),
        'anchor_recovered_from_problem_cluster': recovered_anchors.to_dict(
            orient='records'
        ),
        'candidate_anchor_clusters': anchor_manifest.to_dict(orient='records'),
        'blocked_clusters': blocked_clusters.to_dict(orient='records'),
        'final_verdict': {
            'binary_triage_direction': 'review_prioritization_no_low_vs_moderate_severe',
            'multi_ordinal_status': 'not_primary_product_current_data',
            'claim_boundary': 'Reviewed anchors and score-review evidence support triage/review prioritization only.',
        },
        'next_implementation_step': (
            'Use adjudicated atlas anchors and score-review evidence to evaluate a '
            'binary no/low versus moderate/severe review-triage model with '
            'uncertainty and explanations.'
        ),
    }


def _render_final_adjudication_outcome_md(outcome: dict[str, Any]) -> str:
    lines = [
        '# Final atlas adjudication outcome',
        '',
        f"Status: `{outcome.get('adjudication_status')}`",
        '',
        '## Score Corrections',
    ]
    score_changes = outcome.get('score_changes', [])
    if score_changes:
        for row in score_changes:
            note = f"; note: {row.get('final_notes')}" if row.get('final_notes') else ''
            lines.append(
                f"- Atlas row {row.get('atlas_row_id')} ({row.get('subject_image_id')}): "
                f"{row.get('original_score')} -> {row.get('adjudicated_score')} "
                f"because `{row.get('reason')}`{note}"
            )
    else:
        lines.append('- None')
    lines.extend(['', '## Candidate Anchor Clusters'])
    anchors = outcome.get('candidate_anchor_clusters', [])
    if anchors:
        for row in anchors:
            lines.append(
                f"- Cluster {row.get('cluster_id')}: `{row.get('anchor_class')}`; "
                f"{row.get('accepted_count')}/{row.get('rows_reviewed')} accepted; "
                f"{row.get('cluster_notes')}"
            )
    else:
        lines.append('- None')
    lines.extend(['', '## Recovered Row-Level Anchors'])
    recovered = outcome.get('anchor_recovered_from_problem_cluster', [])
    if recovered:
        for row in recovered:
            lines.append(
                f"- Atlas row {row.get('atlas_row_id')} ({row.get('subject_image_id')}): "
                f"`{row.get('anchor_decision')}` at score {row.get('score')}; "
                f"{row.get('final_notes')}"
            )
    else:
        lines.append('- None')
    lines.extend(['', '## Blocked Clusters'])
    blocked = outcome.get('blocked_clusters', [])
    if blocked:
        for row in blocked:
            lines.append(
                f"- Cluster {row.get('cluster_id')}: `{row.get('cluster_interpretation')}`; "
                f"{row.get('cluster_notes')}"
            )
    else:
        lines.append('- None')
    lines.extend(
        [
            '',
            '## Direction',
            (
                'The repo direction is binary no/low versus moderate/severe review '
                'triage with uncertainty and explanations, not autonomous multi-ordinal '
                'grading.'
            ),
        ]
    )
    return '\n'.join(lines) + '\n'


def _write_flagged_case_review_html(
    atlas_paths: AtlasPaths, case_actions: pd.DataFrame
) -> Path:
    rows = [_flagged_review_payload(row) for _, row in case_actions.iterrows()]
    cards = '\n'.join(_flagged_review_card(row) for row in rows)
    payload = html.escape(json.dumps({'rows': rows}, allow_nan=False), quote=False)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Flagged atlas case review</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1f2933}}
header{{position:sticky;top:0;background:white;border-bottom:1px solid #d8dde6;padding:14px 20px;z-index:10}}
h1{{font-size:22px;margin:0 0 6px}} p{{margin:4px 0;color:#52606d}} main{{padding:18px 20px 40px}}
button{{border:1px solid #9fb3c8;background:white;border-radius:6px;padding:7px 10px;cursor:pointer}}button.primary{{background:#1f5eff;color:white;border-color:#1f5eff}}
.case{{background:white;border:1px solid #d8dde6;border-radius:8px;margin:0 0 16px;overflow:hidden}}
.head{{background:#eef2f7;padding:10px 12px}}.prompt{{padding:10px 12px;border-bottom:1px solid #e4e7eb;font-weight:600}}
.grid{{display:grid;grid-template-columns:1fr 1fr 360px;gap:10px;padding:10px}}figure{{margin:0}}figcaption{{font-size:12px;color:#52606d;margin-bottom:4px}}
img{{width:100%;max-height:360px;object-fit:contain;background:#111827;border-radius:4px}}
.controls{{display:grid;gap:8px;align-content:start}}label{{display:grid;gap:3px;font-size:12px;color:#334e68}}
select,textarea{{width:100%;box-sizing:border-box;border:1px solid #bcccdc;border-radius:5px;padding:6px;font:inherit;background:white}}textarea{{min-height:80px;resize:vertical}}
.meta{{font-size:13px;color:#52606d;line-height:1.45}}.path{{font-size:11px;color:#627d98;overflow-wrap:anywhere}}
@media (max-width: 1000px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Flagged atlas case review</h1><p>Review only these flagged cases. The cases are visible without JavaScript; JavaScript only saves and exports decisions.</p><button class="primary" id="downloadCsv" type="button">Export flagged-case decisions CSV</button> <button id="clearSaved" type="button">Clear saved values</button> <span id="status">Autosaves in this browser.</span></header>
<main id="root">{cards or '<p>No flagged cases selected for focused review.</p>'}</main>
<script id="data" type="application/json">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const key = 'eq.flagged_atlas_case_review.' + location.pathname;
const saved = JSON.parse(localStorage.getItem(key) || '{{}}');
function persist(id, k, v) {{
  saved[id] = {{...(saved[id] || {{}}), [k]: v}};
  localStorage.setItem(key, JSON.stringify(saved));
  document.getElementById('status').textContent = 'Saved locally at ' + new Date().toLocaleTimeString();
}}
document.querySelectorAll('select,textarea').forEach(el => {{
  const state = saved[el.dataset.id] || {{}};
  if (state[el.dataset.key]) el.value = state[el.dataset.key];
  el.addEventListener('change', ev => persist(ev.target.dataset.id, ev.target.dataset.key, ev.target.value));
  el.addEventListener('input', ev => persist(ev.target.dataset.id, ev.target.dataset.key, ev.target.value));
}});
function exportCsv() {{
  const cols = ['review_id','atlas_row_id','subject_image_id','cluster_id','original_score','review_action','score_plausibility','case_cluster_fit','score_decision','corrected_score','score_error_reason','anchor_decision','anchor_exclusion_reason','final_notes','reviewed_at','roi_image_path','roi_mask_path'];
  const rows = data.rows.map(item => {{
    const state = saved[item.review_id] || {{}};
    return {{...item, ...state, reviewed_at: new Date().toISOString()}};
  }});
  const csv = [cols.join(',')].concat(rows.map(r => cols.map(c => `"${{String(r[c] ?? '').replaceAll('"','""')}}"`).join(','))).join('\\n');
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'atlas_flagged_case_decisions.csv'; a.click(); URL.revokeObjectURL(a.href);
}}
document.getElementById('downloadCsv').addEventListener('click', exportCsv);
document.getElementById('clearSaved').addEventListener('click', () => {{ if (confirm('Clear saved decisions?')) {{ localStorage.removeItem(key); location.reload(); }} }});
</script></body></html>"""
    path = atlas_paths.evidence / 'atlas_flagged_case_review.html'
    path.write_text(document, encoding='utf-8')
    return path


def _flagged_review_payload(row: pd.Series) -> dict[str, Any]:
    action = str(row.get('review_action', ''))
    if action == 'exclude_from_anchor':
        fields = {
            'anchor_decision': [
                '',
                'exclude_from_anchor',
                'allow_as_low_anchor',
                'allow_as_borderline_anchor',
                'unclear_needs_second_reviewer',
            ],
            'anchor_exclusion_reason': [
                '',
                'wrong_cluster',
                'atypical_morphology',
                'mixed_features',
                'rbc_confounded',
                'not_representative',
                'other',
            ],
        }
        prompt = 'Confirm whether this case should be excluded from anchor use or recovered as a row-level anchor.'
    else:
        fields = {
            'score_decision': [
                '',
                'keep_original_score',
                'change_score',
                'unclear_needs_second_reviewer',
            ],
            'corrected_score': ['', '0', '0.5', '1', '1.5', '2', '3', 'not_applicable'],
            'score_error_reason': [
                '',
                'rbc_confounded',
                'open_lumina_underrecognized',
                'collapsed_lumina_underrecognized',
                'mixed_features',
                'image_quality_limits',
                'other',
            ],
        }
        prompt = 'Decide whether the original score should stay or change.'
    return {
        'review_id': f"flagged-{row.get('atlas_row_id', '')}",
        'atlas_row_id': str(row.get('atlas_row_id', '')),
        'subject_image_id': str(row.get('subject_image_id', '')),
        'cluster_id': str(row.get('cluster_id', '')),
        'original_score': str(row.get('original_score', '')),
        'review_action': action,
        'score_plausibility': str(row.get('score_plausibility', '')),
        'case_cluster_fit': str(row.get('case_cluster_fit', '')),
        'morphology_assessment': str(row.get('morphology_assessment', '')),
        'reviewer_notes': str(row.get('reviewer_notes', '')),
        'roi_image_path': str(row.get('roi_image_path', '')),
        'roi_mask_path': str(row.get('roi_mask_path', '')),
        'roi_image_src': _review_image_src(row.get('roi_image_path', '')),
        'roi_mask_src': _review_image_src(row.get('roi_mask_path', '')),
        'decision_prompt': prompt,
        'fields': fields,
    }


def _flagged_review_card(row: dict[str, Any]) -> str:
    controls = ''.join(
        _select_control(row['review_id'], field, options)
        for field, options in row['fields'].items()
    )
    return f"""
<article class="case" data-review-id="{html.escape(row['review_id'])}">
  <div class="head"><strong>Atlas row {html.escape(row['atlas_row_id'])} | cluster {html.escape(row['cluster_id'])} | original score {html.escape(row['original_score'])}</strong><div class="meta">{html.escape(row['subject_image_id'])} | action: {html.escape(row['review_action'])} | score plausibility: {html.escape(row['score_plausibility'])} | fit: {html.escape(row['case_cluster_fit'])}</div></div>
  <div class="prompt">{html.escape(row['decision_prompt'])}</div>
  <div class="grid">
    <figure><figcaption>ROI image</figcaption><img src="{html.escape(row['roi_image_src'])}" alt="ROI image for atlas row {html.escape(row['atlas_row_id'])}"></figure>
    <figure><figcaption>ROI mask</figcaption><img src="{html.escape(row['roi_mask_src'])}" alt="ROI mask for atlas row {html.escape(row['atlas_row_id'])}"></figure>
    <div class="controls">{controls}<label>final notes<textarea data-id="{html.escape(row['review_id'])}" data-key="final_notes"></textarea></label><div class="meta"><strong>Prior morphology:</strong> {html.escape(row['morphology_assessment'])}<br><strong>Prior note:</strong> {html.escape(row['reviewer_notes'])}</div><div class="path">Image: {html.escape(row['roi_image_path'])}<br>Mask: {html.escape(row['roi_mask_path'])}</div></div>
  </div>
</article>"""


def _select_control(review_id: str, field: str, options: list[str]) -> str:
    choices = ''.join(
        f'<option value="{html.escape(value)}">{html.escape(value or "choose")}</option>'
        for value in options
    )
    return (
        f'<label>{html.escape(field.replace("_", " "))}<select '
        f'data-id="{html.escape(review_id)}" data-key="{html.escape(field)}">'
        f'{choices}</select></label>'
    )


def _review_image_src(value: Any) -> str:
    text = str(value or '')
    if not text:
        return ''
    path = Path(text)
    if path.exists():
        return path.as_uri()
    return text


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


def _write_binary_review_triage_outputs(
    *,
    atlas_paths: AtlasPaths,
    inputs: AtlasInputs,
    assignments: pd.DataFrame,
    feature_spaces: list[FeatureSpace],
    anchor_manifest: pd.DataFrame,
    blocked_clusters: pd.DataFrame,
    recovered_anchors: pd.DataFrame,
    score_corrections: pd.DataFrame,
) -> dict[str, Path]:
    triage_paths = _binary_triage_paths(atlas_paths)
    for path in triage_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    metadata = inputs.embeddings.reset_index(drop=True).copy()
    metadata['atlas_row_id'] = np.arange(len(metadata), dtype=int)
    primary = _primary_assignments(assignments)
    frame = _binary_triage_frame(
        metadata=metadata,
        primary=primary,
        feature_spaces=feature_spaces,
        anchor_manifest=anchor_manifest,
        blocked_clusters=blocked_clusters,
        recovered_anchors=recovered_anchors,
        score_corrections=score_corrections,
    )
    frame = _add_binary_targets(frame)
    feature_sets = _binary_feature_sets(frame)

    changed: dict[str, Path] = {}
    metrics_rows: list[dict[str, Any]] = []
    oof_predictions: dict[tuple[str, str], pd.DataFrame] = {}

    baseline_row, baseline_oof = _evaluate_cluster_mapping_candidate(frame)
    metrics_rows.append(baseline_row)
    oof_predictions[(baseline_row['candidate_id'], baseline_row['target_kind'])] = (
        baseline_oof
    )

    for target_kind, target_column in [
        ('primary_no_low_vs_moderate_severe', 'binary_target_primary_value'),
        ('sensitivity_no_low_inclusive', 'binary_target_sensitivity_value'),
    ]:
        for candidate_id, payload in feature_sets.items():
            row, oof = _evaluate_binary_logistic_candidate(
                frame,
                candidate_id=candidate_id,
                feature_family=str(payload['feature_family']),
                feature_columns=list(payload['columns']),
                target_kind=target_kind,
                target_column=target_column,
            )
            metrics_rows.append(row)
            oof_predictions[(candidate_id, target_kind)] = oof

    metrics = pd.DataFrame(metrics_rows)
    selected = _select_binary_candidate(metrics)
    predictions, explanations, model_support, model_path = _binary_triage_predictions(
        frame=frame,
        feature_sets=feature_sets,
        selected=selected,
        oof_predictions=oof_predictions,
        output_dir=triage_paths['model'],
    )
    intervals = _binary_metric_intervals(
        frame=frame,
        selected=selected,
        oof_predictions=oof_predictions,
    )
    verdict = _binary_triage_verdict(
        frame=frame,
        selected=selected,
        metrics=metrics,
        intervals=intervals,
        model_support=model_support,
        anchor_manifest=anchor_manifest,
        blocked_clusters=blocked_clusters,
        recovered_anchors=recovered_anchors,
    )

    predictions_path = triage_paths['predictions'] / 'binary_triage_predictions.csv'
    predictions.to_csv(predictions_path, index=False)
    changed['binary_triage_predictions'] = predictions_path

    explanation_path = triage_paths['evidence'] / 'binary_triage_explanations.csv'
    explanations.to_csv(explanation_path, index=False)
    changed['binary_triage_explanations'] = explanation_path

    metrics_path = triage_paths['summary'] / 'binary_triage_metrics.csv'
    metrics.to_csv(metrics_path, index=False)
    changed['binary_triage_metrics'] = metrics_path

    intervals_path = save_json(
        _json_ready(intervals),
        triage_paths['summary'] / 'binary_triage_metric_intervals.json',
    )
    changed['binary_triage_metric_intervals'] = intervals_path

    support_path = save_json(
        _json_ready(model_support),
        triage_paths['diagnostics'] / 'binary_triage_support.json',
    )
    changed['binary_triage_support'] = support_path

    verdict_json_path = save_json(
        _json_ready(verdict), triage_paths['summary'] / 'binary_triage_verdict.json'
    )
    verdict_md_path = triage_paths['summary'] / 'binary_triage_verdict.md'
    verdict_md_path.write_text(_render_binary_triage_verdict_md(verdict), encoding='utf-8')
    changed['binary_triage_verdict'] = verdict_json_path
    changed['binary_triage_verdict_md'] = verdict_md_path

    review_path = _write_binary_triage_review_html(
        triage_paths['evidence'], predictions, explanations
    )
    changed['binary_triage_review_html'] = review_path

    if model_path is not None:
        changed['binary_triage_selected_model'] = model_path
        model_manifest_path = save_json(
            _json_ready(
                {
                    'model_path': str(model_path),
                    'storage_policy': (
                        'runtime_artifact_not_committed; use Git LFS only after a '
                        'separate promotion gate names this as a stable model'
                    ),
                    'selected_candidate_id': selected.get('candidate_id', ''),
                    'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
                }
            ),
            triage_paths['model'] / 'model_manifest.json',
        )
        changed['binary_triage_model_manifest'] = model_manifest_path

    index_path = triage_paths['root'] / 'INDEX.md'
    index_path.write_text(_render_binary_triage_index(verdict), encoding='utf-8')
    changed['binary_triage_index'] = index_path
    return changed


def _binary_triage_paths(atlas_paths: AtlasPaths) -> dict[str, Path]:
    root = atlas_paths.root / BINARY_TRIAGE_SUBTREE
    return {
        'root': root,
        'summary': root / 'summary',
        'predictions': root / 'predictions',
        'evidence': root / 'evidence',
        'diagnostics': root / 'diagnostics',
        'model': root / 'model',
    }


def _binary_triage_frame(
    *,
    metadata: pd.DataFrame,
    primary: pd.DataFrame,
    feature_spaces: list[FeatureSpace],
    anchor_manifest: pd.DataFrame,
    blocked_clusters: pd.DataFrame,
    recovered_anchors: pd.DataFrame,
    score_corrections: pd.DataFrame,
) -> pd.DataFrame:
    frame = metadata.copy()
    primary_keep = [
        column
        for column in [
            'atlas_row_id',
            'feature_space_id',
            'method_id',
            'cluster_id',
            'assignment_confidence',
            'assignment_distance',
        ]
        if column in primary.columns
    ]
    if primary_keep:
        frame = frame.merge(
            primary.loc[:, primary_keep].drop_duplicates('atlas_row_id'),
            on='atlas_row_id',
            how='left',
        )
    frame['selected_cluster_id'] = frame.get('cluster_id', '').fillna('').astype(str)
    frame['selected_view_id'] = (
        frame.get('feature_space_id', '').fillna('').astype(str)
        + '/'
        + frame.get('method_id', '').fillna('').astype(str)
    )
    frame['blocked_cluster_indicator'] = _blocked_cluster_indicator(
        frame, blocked_clusters
    )
    frame['score_correction_indicator'] = frame['atlas_row_id'].astype(str).isin(
        set(score_corrections.get('atlas_row_id', pd.Series(dtype=str)).astype(str))
    )
    anchor_records = _anchor_records(anchor_manifest, recovered_anchors)
    nearest_anchor = _nearest_anchor_frame(frame, feature_spaces, anchor_records)
    if not nearest_anchor.empty:
        frame = frame.merge(nearest_anchor, on='atlas_row_id', how='left')
    for column in [
        'nearest_anchor_atlas_row_id',
        'nearest_anchor_class',
        'nearest_anchor_distance',
        'nearest_anchor_source',
    ]:
        if column not in frame:
            frame[column] = ''
    return frame


def _blocked_cluster_indicator(
    frame: pd.DataFrame, blocked_clusters: pd.DataFrame
) -> pd.Series:
    if blocked_clusters.empty:
        return pd.Series(False, index=frame.index)
    keys = set(
        zip(
            blocked_clusters.get('feature_space_id', pd.Series(dtype=str)).astype(str),
            blocked_clusters.get('method_id', pd.Series(dtype=str)).astype(str),
            blocked_clusters.get('cluster_id', pd.Series(dtype=str)).astype(str),
        )
    )
    row_keys = zip(
        frame.get('feature_space_id', pd.Series('', index=frame.index)).astype(str),
        frame.get('method_id', pd.Series('', index=frame.index)).astype(str),
        frame.get('cluster_id', pd.Series('', index=frame.index)).astype(str),
    )
    return pd.Series([key in keys for key in row_keys], index=frame.index)


def _anchor_records(
    anchor_manifest: pd.DataFrame, recovered_anchors: pd.DataFrame
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not anchor_manifest.empty:
        for _, row in anchor_manifest.iterrows():
            anchor_class = str(row.get('anchor_class', ''))
            for value in str(row.get('atlas_row_ids', '')).split(';'):
                if value.strip():
                    records.append(
                        {
                            'atlas_row_id': int(value),
                            'anchor_class': anchor_class,
                            'anchor_source': 'cluster_manifest',
                        }
                    )
    if not recovered_anchors.empty:
        for _, row in recovered_anchors.iterrows():
            decision = str(row.get('anchor_decision', ''))
            anchor_class = (
                'no_low'
                if 'low' in decision
                else 'borderline_review'
                if 'borderline' in decision
                else ''
            )
            if anchor_class:
                records.append(
                    {
                        'atlas_row_id': int(float(row.get('atlas_row_id'))),
                        'anchor_class': anchor_class,
                        'anchor_source': 'row_recovered_anchor',
                    }
                )
    return records


def _nearest_anchor_frame(
    frame: pd.DataFrame,
    feature_spaces: list[FeatureSpace],
    anchor_records: list[dict[str, Any]],
) -> pd.DataFrame:
    if not anchor_records or not feature_spaces:
        return pd.DataFrame()
    preferred = next(
        (space for space in feature_spaces if space.space_id == 'encoder_pca'),
        feature_spaces[0],
    )
    anchor_df = pd.DataFrame(anchor_records).drop_duplicates('atlas_row_id')
    valid_anchor_ids = [
        int(value)
        for value in anchor_df['atlas_row_id'].tolist()
        if 0 <= int(value) < preferred.matrix.shape[0]
    ]
    if not valid_anchor_ids:
        return pd.DataFrame()
    anchor_lookup = anchor_df.set_index('atlas_row_id').to_dict('index')
    anchor_matrix = preferred.matrix[valid_anchor_ids]
    rows: list[dict[str, Any]] = []
    for atlas_row_id in frame['atlas_row_id'].astype(int):
        vector = preferred.matrix[int(atlas_row_id)]
        distances = np.linalg.norm(anchor_matrix - vector, axis=1)
        order = np.argsort(distances)
        chosen_order = int(order[0])
        chosen_id = int(valid_anchor_ids[chosen_order])
        if chosen_id == int(atlas_row_id) and len(order) > 1:
            chosen_order = int(order[1])
            chosen_id = int(valid_anchor_ids[chosen_order])
        payload = anchor_lookup.get(chosen_id, {})
        rows.append(
            {
                'atlas_row_id': int(atlas_row_id),
                'nearest_anchor_atlas_row_id': chosen_id,
                'nearest_anchor_class': payload.get('anchor_class', ''),
                'nearest_anchor_distance': float(distances[chosen_order]),
                'nearest_anchor_source': payload.get('anchor_source', ''),
            }
        )
    return pd.DataFrame(rows)


def _add_binary_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    score = pd.to_numeric(out.get('score'), errors='coerce')
    out['binary_target_primary'] = np.select(
        [score <= 0.5, score >= 1.5, score.eq(1.0)],
        ['no_low', 'moderate_severe', 'borderline_review'],
        default='unscored',
    )
    out['binary_target_primary_value'] = np.where(
        score <= 0.5, 0, np.where(score >= 1.5, 1, np.nan)
    )
    out['binary_target_sensitivity'] = np.select(
        [score <= 1.0, score >= 1.5], ['no_low_inclusive', 'moderate_severe'], default='unscored'
    )
    out['binary_target_sensitivity_value'] = np.where(
        score <= 1.0, 0, np.where(score >= 1.5, 1, np.nan)
    )
    return out


def _binary_feature_sets(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    work = frame.copy()
    for column in ['selected_cluster_id', 'nearest_anchor_class']:
        if column in work:
            dummies = pd.get_dummies(work[column].fillna('').astype(str), prefix=column)
            for dummy in dummies.columns:
                frame[dummy] = dummies[dummy].astype(float)
    frame['blocked_cluster_numeric'] = frame.get(
        'blocked_cluster_indicator', pd.Series(False, index=frame.index)
    ).astype(float)
    frame['score_correction_numeric'] = frame.get(
        'score_correction_indicator', pd.Series(False, index=frame.index)
    ).astype(float)
    numeric_columns = set(frame.select_dtypes(include=[np.number, bool]).columns)
    embedding_columns = [
        column for column in _encoder_columns(frame) if column in numeric_columns
    ]
    roi_columns = [column for column in _roi_qc_columns(frame) if column in numeric_columns]
    cluster_columns = [
        column
        for column in frame.columns
        if column.startswith('selected_cluster_id_')
        or column.startswith('nearest_anchor_class_')
        or column
        in {
            'assignment_confidence',
            'assignment_distance',
            'nearest_anchor_distance',
            'blocked_cluster_numeric',
            'score_correction_numeric',
        }
    ]
    learned_columns = _learned_feature_columns(frame)
    return {
        'roi_qc_binary_logistic': {
            'feature_family': 'roi_qc',
            'columns': roi_columns,
        },
        'embedding_binary_logistic': {
            'feature_family': 'embedding',
            'columns': embedding_columns,
        },
        'atlas_hybrid_binary_logistic': {
            'feature_family': 'embedding_roi_qc_atlas_anchor',
            'columns': sorted(
                set(embedding_columns + roi_columns + cluster_columns + learned_columns)
            ),
        },
    }


def _evaluate_cluster_mapping_candidate(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_column = 'binary_target_primary_value'
    target_mask = pd.to_numeric(frame[target_column], errors='coerce').notna()
    work = frame[target_mask].copy()
    if work.empty:
        return _non_estimable_metric_row(
            candidate_id='atlas_cluster_anchor_mapping',
            target_kind='primary_no_low_vs_moderate_severe',
            feature_family='atlas_anchor_cluster',
            model_kind='rule_mapping',
            reason='no_primary_binary_target_rows',
            row_count=0,
            subject_count=0,
        ), pd.DataFrame()
    cluster_prob = np.full(len(work), np.nan)
    class_by_cluster: dict[str, int] = {}
    grouped = work.groupby('selected_cluster_id', dropna=False)
    for cluster_id, group in grouped:
        labels = pd.to_numeric(group[target_column], errors='coerce').dropna()
        if labels.empty:
            continue
        class_by_cluster[str(cluster_id)] = int(labels.mean() >= 0.5)
    for pos, (_, row) in enumerate(work.iterrows()):
        if bool(row.get('blocked_cluster_indicator', False)):
            continue
        mapped = class_by_cluster.get(str(row.get('selected_cluster_id', '')))
        if mapped is not None:
            cluster_prob[pos] = float(mapped)
    finite = np.isfinite(cluster_prob)
    oof = pd.DataFrame(
        {
            'atlas_row_id': work['atlas_row_id'].to_numpy(dtype=int),
            'target': pd.to_numeric(work[target_column], errors='coerce').to_numpy(),
            'probability': cluster_prob,
            'subject_id': work.get('subject_id', '').astype(str).to_numpy(),
        }
    )
    if finite.sum() == 0:
        return _non_estimable_metric_row(
            candidate_id='atlas_cluster_anchor_mapping',
            target_kind='primary_no_low_vs_moderate_severe',
            feature_family='atlas_anchor_cluster',
            model_kind='rule_mapping',
            reason='no_unblocked_cluster_mapping',
            row_count=int(len(work)),
            subject_count=_nunique(work, 'subject_id'),
        ), oof
    metrics = _binary_metric_values(
        y_true=oof.loc[finite, 'target'].to_numpy(dtype=int),
        probability=oof.loc[finite, 'probability'].to_numpy(dtype=float),
        threshold=0.5,
    )
    return {
        'candidate_id': 'atlas_cluster_anchor_mapping',
        'target_kind': 'primary_no_low_vs_moderate_severe',
        'feature_family': 'atlas_anchor_cluster',
        'model_kind': 'blocked_clusters_left_unassigned_rule_mapping',
        'metric_label': 'grouped_out_of_fold_development_estimate',
        'finite_output_status': 'finite' if metrics['finite'] else 'non_estimable',
        'hard_blockers': '',
        'threshold': 0.5,
        'threshold_selection_objective': 'fixed_cluster_mapping_probability_0_5',
        'row_count': int(finite.sum()),
        'subject_count': int(
            work.loc[finite, 'subject_id'].astype(str).nunique()
            if 'subject_id' in work
            else 0
        ),
        **{key: value for key, value in metrics.items() if key != 'finite'},
    }, oof


def _evaluate_binary_logistic_candidate(
    frame: pd.DataFrame,
    *,
    candidate_id: str,
    feature_family: str,
    feature_columns: list[str],
    target_kind: str,
    target_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = pd.to_numeric(frame.get(target_column), errors='coerce')
    mask = target.notna()
    work = frame.loc[mask].copy()
    if not feature_columns:
        return _non_estimable_metric_row(
            candidate_id=candidate_id,
            target_kind=target_kind,
            feature_family=feature_family,
            model_kind='subject_grouped_logistic_regression',
            reason='no_feature_columns',
            row_count=int(len(work)),
            subject_count=_nunique(work, 'subject_id'),
        ), pd.DataFrame()
    if work.empty or target[mask].nunique() < 2:
        return _non_estimable_metric_row(
            candidate_id=candidate_id,
            target_kind=target_kind,
            feature_family=feature_family,
            model_kind='subject_grouped_logistic_regression',
            reason='binary_target_not_estimable',
            row_count=int(len(work)),
            subject_count=_nunique(work, 'subject_id'),
        ), pd.DataFrame()

    y = pd.to_numeric(work[target_column], errors='coerce').astype(int).to_numpy()
    groups = work.get('subject_id', pd.Series(np.arange(len(work)), index=work.index))
    groups = groups.astype(str).to_numpy()
    oof_probability = np.full(len(work), np.nan, dtype=float)
    skipped: list[str] = []
    numeric_warning_messages: list[str] = []
    for train_idx, test_idx in _subject_grouped_splits(y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            skipped.append('single_class_fold')
            continue
        model, fit_warnings = _capture_runtime_warnings(
            lambda: _fit_binary_logistic(work.iloc[train_idx], y[train_idx], feature_columns)
        )
        numeric_warning_messages.extend(fit_warnings)
        if not _binary_model_finite(model):
            skipped.append('nonfinite_model_coefficients')
            continue
        fold_probability, predict_warnings = _capture_runtime_warnings(
            lambda: model.predict_proba(
                _feature_matrix(work.iloc[test_idx], feature_columns)
            )[:, 1]
        )
        numeric_warning_messages.extend(predict_warnings)
        if not np.isfinite(fold_probability).all():
            skipped.append('nonfinite_fold_probabilities')
            continue
        oof_probability[test_idx] = fold_probability
    finite = np.isfinite(oof_probability)
    oof = pd.DataFrame(
        {
            'atlas_row_id': work['atlas_row_id'].to_numpy(dtype=int),
            'target': y,
            'probability': oof_probability,
            'subject_id': groups,
        }
    )
    if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
        return _non_estimable_metric_row(
            candidate_id=candidate_id,
            target_kind=target_kind,
            feature_family=feature_family,
            model_kind='subject_grouped_logistic_regression',
            reason='no_estimable_grouped_oof_predictions',
            row_count=int(len(work)),
            subject_count=_nunique(work, 'subject_id'),
            extra={'skipped_fold_reasons': ';'.join(sorted(set(skipped)))},
        ), oof
    threshold, threshold_objective = _select_binary_operating_threshold(
        y_true=y[finite],
        probability=oof_probability[finite],
    )
    metrics = _binary_metric_values(
        y_true=y[finite],
        probability=oof_probability[finite],
        threshold=threshold,
    )
    return {
        'candidate_id': candidate_id,
        'target_kind': target_kind,
        'feature_family': feature_family,
        'model_kind': 'subject_grouped_logistic_regression',
        'metric_label': 'grouped_out_of_fold_development_estimate',
        'finite_output_status': 'finite' if metrics['finite'] else 'non_estimable',
        'hard_blockers': '',
        'threshold': threshold,
        'threshold_selection_objective': threshold_objective,
        'row_count': int(finite.sum()),
        'subject_count': int(len(set(groups[finite]))),
        'feature_count': int(len(feature_columns)),
        'skipped_fold_reasons': ';'.join(sorted(set(skipped))),
        'numeric_warning_count': len(numeric_warning_messages),
        'numeric_warning_messages': '; '.join(
            list(dict.fromkeys(numeric_warning_messages))[:3]
        ),
        **{key: value for key, value in metrics.items() if key != 'finite'},
    }, oof


def _subject_grouped_splits(
    target: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    class_counts = pd.Series(target).value_counts()
    n_splits = min(5, len(unique_groups), int(class_counts.min()))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(np.zeros(len(target)), target, groups)
    ]


def _fit_binary_logistic(frame: pd.DataFrame, target: np.ndarray, columns: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ('scaler', StandardScaler()),
            (
                'logistic',
                LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=0,
                    solver='liblinear',
                ),
            ),
        ]
    )
    model.fit(_feature_matrix(frame, columns), target)
    return model


def _binary_model_finite(model: Any) -> bool:
    try:
        logistic = model.named_steps['logistic']
    except Exception:
        return False
    coef = np.asarray(getattr(logistic, 'coef_', []), dtype=float)
    intercept = np.asarray(getattr(logistic, 'intercept_', []), dtype=float)
    return bool(np.isfinite(coef).all() and np.isfinite(intercept).all())


def _feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return to_finite_numeric_matrix(frame, columns, finite_bound=1e6)


def _binary_metric_values(
    *, y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    predicted = probability >= threshold
    y_bool = y_true.astype(bool)
    tp = int(np.logical_and(predicted, y_bool).sum())
    tn = int(np.logical_and(~predicted, ~y_bool).sum())
    fp = int(np.logical_and(predicted, ~y_bool).sum())
    fn = int(np.logical_and(~predicted, y_bool).sum())
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    auroc = None
    average_precision = None
    if len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, probability))
        average_precision = float(average_precision_score(y_true, probability))
    return {
        'finite': balanced is not None,
        'recall': _optional_float(recall),
        'precision': _optional_float(precision),
        'specificity': _optional_float(specificity),
        'balanced_accuracy': _optional_float(balanced),
        'auroc': _optional_float(auroc),
        'average_precision': _optional_float(average_precision),
        'false_negative_count': fn,
        'false_positive_count': fp,
        'true_positive_count': tp,
        'true_negative_count': tn,
    }


def _select_binary_operating_threshold(
    *, y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, str]:
    objective = 'maximize_grouped_oof_balanced_accuracy_then_moderate_severe_recall'
    candidates = np.round(np.linspace(0.05, 0.95, 19), 2)
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0, -abs(0.5 - best_threshold))
    for threshold in candidates:
        metrics = _binary_metric_values(
            y_true=y_true,
            probability=probability,
            threshold=float(threshold),
        )
        key = (
            float(metrics.get('balanced_accuracy') or -1.0),
            float(metrics.get('recall') or -1.0),
            float(metrics.get('specificity') or -1.0),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, objective


def _non_estimable_metric_row(
    *,
    candidate_id: str,
    target_kind: str,
    feature_family: str,
    model_kind: str,
    reason: str,
    row_count: int,
    subject_count: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'candidate_id': candidate_id,
        'target_kind': target_kind,
        'feature_family': feature_family,
        'model_kind': model_kind,
        'metric_label': 'grouped_out_of_fold_development_estimate',
        'finite_output_status': 'non_estimable',
        'hard_blockers': reason,
        'threshold': 0.5,
        'threshold_selection_objective': 'non_estimable',
        'row_count': int(row_count),
        'subject_count': int(subject_count),
        'recall': None,
        'precision': None,
        'specificity': None,
        'balanced_accuracy': None,
        'auroc': None,
        'average_precision': None,
        'false_negative_count': None,
        'false_positive_count': None,
        **(extra or {}),
    }


def _select_binary_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    primary = metrics[
        metrics['target_kind'].eq('primary_no_low_vs_moderate_severe')
        & metrics['finite_output_status'].eq('finite')
        & metrics['model_kind'].astype(str).str.contains('logistic')
    ].copy()
    if primary.empty:
        return {
            'candidate_id': '',
            'target_kind': 'primary_no_low_vs_moderate_severe',
            'feature_family': '',
            'threshold': 0.5,
            'selection_status': 'no_estimable_binary_candidate',
            'selection_rule': 'balanced_accuracy_then_average_precision',
        }
    primary['balanced_accuracy_sort'] = pd.to_numeric(
        primary['balanced_accuracy'], errors='coerce'
    ).fillna(-1.0)
    primary['average_precision_sort'] = pd.to_numeric(
        primary['average_precision'], errors='coerce'
    ).fillna(-1.0)
    primary['hybrid_bonus'] = primary['candidate_id'].eq(
        'atlas_hybrid_binary_logistic'
    ).astype(int)
    row = primary.sort_values(
        ['balanced_accuracy_sort', 'average_precision_sort', 'hybrid_bonus'],
        ascending=False,
    ).iloc[0]
    return {
        **row.to_dict(),
        'selection_status': 'selected_current_data_review_triage_candidate',
        'selection_rule': 'balanced_accuracy_then_average_precision',
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _binary_triage_predictions(
    *,
    frame: pd.DataFrame,
    feature_sets: dict[str, dict[str, Any]],
    selected: dict[str, Any],
    oof_predictions: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], Path | None]:
    candidate_id = str(selected.get('candidate_id') or '')
    target_kind = str(selected.get('target_kind') or 'primary_no_low_vs_moderate_severe')
    threshold = float(selected.get('threshold') or 0.5)
    feature_payload = feature_sets.get(candidate_id, {})
    feature_columns = list(feature_payload.get('columns') or [])
    target = pd.to_numeric(frame.get('binary_target_primary_value'), errors='coerce')
    train_mask = target.notna()
    model_path: Path | None = None
    model_support = {
        'selected_candidate_id': candidate_id,
        'selection_status': selected.get('selection_status', ''),
        'feature_columns': feature_columns,
        'primary_target_rows': int(train_mask.sum()),
        'primary_target_subjects': int(
            frame.loc[train_mask, 'subject_id'].astype(str).nunique()
            if 'subject_id' in frame
            else 0
        ),
        'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
    }
    if not candidate_id or not feature_columns or target[train_mask].nunique() < 2:
        probability = np.full(len(frame), 0.5)
        model_support['finite_model_status'] = 'not_fit_insufficient_grouped_support'
        model = None
    else:
        y = target[train_mask].astype(int).to_numpy()
        model, fit_warnings = _capture_runtime_warnings(
            lambda: _fit_binary_logistic(frame.loc[train_mask], y, feature_columns)
        )
        if not _binary_model_finite(model):
            probability = np.full(len(frame), 0.5)
            model_support['finite_model_status'] = 'not_fit_nonfinite_coefficients'
            model_support['numeric_warning_count'] = len(fit_warnings)
            model_support['numeric_warning_messages'] = fit_warnings[:3]
            model = None
        else:
            probability, predict_warnings = _capture_runtime_warnings(
                lambda: model.predict_proba(_feature_matrix(frame, feature_columns))[:, 1]
            )
            if not np.isfinite(probability).all():
                probability = np.full(len(frame), 0.5)
                model_support['finite_model_status'] = 'not_fit_nonfinite_probabilities'
                model = None
            else:
                model_support['finite_model_status'] = 'fit_on_all_primary_target_rows'
                model_path = save_supported_sklearn_model(
                    model, output_dir / 'binary_triage_selected_model.joblib'
                )
            all_warnings = fit_warnings + predict_warnings
            model_support['numeric_warning_count'] = len(all_warnings)
            model_support['numeric_warning_messages'] = list(
                dict.fromkeys(all_warnings)
            )[:3]

    oof = oof_predictions.get((candidate_id, target_kind), pd.DataFrame())
    oof_lookup = (
        oof.dropna(subset=['probability']).set_index('atlas_row_id')['probability'].to_dict()
        if not oof.empty and 'probability' in oof
        else {}
    )
    predictions = frame.copy()
    predictions['predicted_probability_moderate_severe'] = probability
    predictions['grouped_oof_probability_moderate_severe'] = predictions[
        'atlas_row_id'
    ].map(oof_lookup)
    predictions['threshold'] = threshold
    predictions['binary_decision'] = np.where(
        predictions['predicted_probability_moderate_severe'] >= threshold,
        'moderate_severe',
        'no_low',
    )
    distance = (
        predictions['predicted_probability_moderate_severe'].astype(float) - threshold
    ).abs()
    predictions['near_threshold'] = distance <= 0.15
    predictions['uncertainty_label'] = np.select(
        [
            predictions['near_threshold'],
            distance <= 0.25,
            distance > 0.35,
        ],
        ['near_threshold_review', 'moderate_uncertainty', 'higher_margin'],
        default='uncertain',
    )
    predictions['source_warning'] = np.where(
        predictions.get('blocked_cluster_indicator', False).astype(bool),
        'blocked_or_mixed_cluster',
        '',
    )
    predictions['source_cohort_warning'] = predictions['source_warning']
    predictions['final_review_route'] = _binary_review_routes(predictions)

    explanation = _binary_explanations(
        frame=frame,
        predictions=predictions,
        model=model,
        feature_columns=feature_columns,
    )
    keep = [
        'atlas_row_id',
        'subject_id',
        'subject_image_id',
        'glomerulus_id',
        'score',
        'binary_target_primary',
        'binary_target_sensitivity',
        'predicted_probability_moderate_severe',
        'grouped_oof_probability_moderate_severe',
        'threshold',
        'binary_decision',
        'uncertainty_label',
        'near_threshold',
        'source_warning',
        'source_cohort_warning',
        'nearest_anchor_atlas_row_id',
        'nearest_anchor_class',
        'nearest_anchor_distance',
        'final_review_route',
        'roi_image_path',
        'roi_mask_path',
    ]
    for column in keep:
        if column not in predictions:
            predictions[column] = ''
    return predictions.loc[:, keep], explanation, model_support, model_path


def _binary_review_routes(predictions: pd.DataFrame) -> np.ndarray:
    score = pd.to_numeric(predictions.get('score'), errors='coerce')
    probability = predictions['predicted_probability_moderate_severe'].astype(float)
    blocked = predictions.get('blocked_cluster_indicator', False).astype(bool)
    near = predictions['near_threshold'].astype(bool)
    routes = np.where(
        score.eq(1.0),
        'borderline_score_review',
        np.where(
            blocked,
            'blocked_cluster_manual_review',
            np.where(
                near,
                'uncertain_binary_review',
                np.where(
                    probability >= 0.65,
                    'likely_moderate_severe_review',
                    np.where(
                        probability <= 0.35,
                        'likely_no_low_review',
                        'uncertain_binary_review',
                    ),
                ),
            ),
        ),
    )
    return routes


def _binary_explanations(
    *,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    columns = [
        'atlas_row_id',
        'top_feature_contributions',
        'feature_family_contributions',
        'top_positive_features',
        'top_negative_features',
        'roi_qc_contribution',
        'embedding_contribution',
        'atlas_cluster_contribution',
        'anchor_distance_contribution',
        'learned_roi_contribution',
        'explanation_claim_boundary',
    ]
    if model is None or not feature_columns:
        out = predictions[['atlas_row_id']].copy()
        for column in columns:
            if column not in out:
                out[column] = ''
        out['explanation_claim_boundary'] = (
            'No fitted model; no row-level feature explanation is estimable.'
        )
        return out.loc[:, columns]
    scaler = model.named_steps['scaler']
    logistic = model.named_steps['logistic']
    transformed = scaler.transform(_feature_matrix(frame, feature_columns))
    coef = logistic.coef_.reshape(-1)
    contributions = transformed * coef
    rows: list[dict[str, Any]] = []
    for pos, atlas_row_id in enumerate(predictions['atlas_row_id'].astype(int)):
        contrib = contributions[pos]
        order = np.argsort(contrib)
        positive = [
            f'{feature_columns[index]}={contrib[index]:.4g}'
            for index in order[::-1][:3]
        ]
        negative = [
            f'{feature_columns[index]}={contrib[index]:.4g}' for index in order[:3]
        ]
        family = _feature_family_contributions(feature_columns, contrib)
        family_text = '; '.join(f'{key}={value:.4g}' for key, value in family.items())
        rows.append(
            {
                'atlas_row_id': int(atlas_row_id),
                'top_feature_contributions': (
                    f"positive: {'; '.join(positive)} | negative: {'; '.join(negative)}"
                ),
                'feature_family_contributions': family_text,
                'top_positive_features': '; '.join(positive),
                'top_negative_features': '; '.join(negative),
                'roi_qc_contribution': family['roi_qc'],
                'embedding_contribution': family['embedding'],
                'atlas_cluster_contribution': family['atlas_cluster'],
                'anchor_distance_contribution': family['anchor_distance'],
                'learned_roi_contribution': family['learned_roi'],
                'explanation_claim_boundary': (
                    'Feature contributions describe the fitted review-triage '
                    'decision surface, not biological causality.'
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _feature_family_contributions(
    feature_columns: list[str], contributions: np.ndarray
) -> dict[str, float]:
    totals = {
        'roi_qc': 0.0,
        'embedding': 0.0,
        'atlas_cluster': 0.0,
        'anchor_distance': 0.0,
        'learned_roi': 0.0,
    }
    for column, contribution in zip(feature_columns, contributions):
        family = 'atlas_cluster'
        if column.startswith('embedding_'):
            family = 'embedding'
        elif column in ROI_QC_COLUMNS:
            family = 'roi_qc'
        elif column.startswith('learned_'):
            family = 'learned_roi'
        elif 'anchor' in column:
            family = 'anchor_distance'
        totals[family] += float(contribution)
    return {key: round(value, 6) for key, value in totals.items()}


def _binary_metric_intervals(
    *,
    frame: pd.DataFrame,
    selected: dict[str, Any],
    oof_predictions: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    candidate_id = str(selected.get('candidate_id') or '')
    target_kind = str(selected.get('target_kind') or 'primary_no_low_vs_moderate_severe')
    oof = oof_predictions.get((candidate_id, target_kind), pd.DataFrame())
    if oof.empty or oof['probability'].dropna().empty:
        return {
            'status': 'non_estimable',
            'reason': 'selected_candidate_has_no_grouped_oof_predictions',
            'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
        }
    work = oof.dropna(subset=['probability']).copy()
    if work['subject_id'].astype(str).nunique() < 3:
        return {
            'status': 'non_estimable',
            'reason': 'fewer_than_three_subjects_for_grouped_bootstrap',
            'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
        }
    rng = np.random.default_rng(0)
    subjects = work['subject_id'].astype(str).unique()
    values: dict[str, list[float]] = {
        key: []
        for key in ['recall', 'precision', 'specificity', 'balanced_accuracy', 'auroc']
    }
    for _ in range(200):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_rows = pd.concat(
            [work[work['subject_id'].astype(str).eq(subject)] for subject in sampled],
            ignore_index=True,
        )
        if sampled_rows['target'].nunique() < 2:
            continue
        metric = _binary_metric_values(
            y_true=sampled_rows['target'].to_numpy(dtype=int),
            probability=sampled_rows['probability'].to_numpy(dtype=float),
            threshold=float(selected.get('threshold') or 0.5),
        )
        for key in values:
            if metric.get(key) is not None:
                values[key].append(float(metric[key]))
    intervals = {}
    for key, series in values.items():
        if series:
            intervals[key] = {
                'estimate': _optional_float(selected.get(key)),
                'ci_method': 'subject_grouped_bootstrap_percentile',
                'ci_lower': float(np.percentile(series, 2.5)),
                'ci_upper': float(np.percentile(series, 97.5)),
                'resamples': int(len(series)),
            }
        else:
            intervals[key] = {
                'estimate': _optional_float(selected.get(key)),
                'ci_method': 'subject_grouped_bootstrap_percentile',
                'non_estimable_reason': 'resamples_lacked_both_classes',
                'resamples': 0,
            }
    return {
        'status': 'estimated' if any(item.get('resamples', 0) for item in intervals.values()) else 'non_estimable',
        'candidate_id': candidate_id,
        'target_kind': target_kind,
        'intervals': intervals,
        'row_count': int(len(work)),
        'subject_count': int(len(subjects)),
        'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
    }


def _binary_triage_verdict(
    *,
    frame: pd.DataFrame,
    selected: dict[str, Any],
    metrics: pd.DataFrame,
    intervals: dict[str, Any],
    model_support: dict[str, Any],
    anchor_manifest: pd.DataFrame,
    blocked_clusters: pd.DataFrame,
    recovered_anchors: pd.DataFrame,
) -> dict[str, Any]:
    primary = frame['binary_target_primary'].astype(str).value_counts().to_dict()
    sensitivity = frame['binary_target_sensitivity'].astype(str).value_counts().to_dict()
    finite_candidates = int(metrics['finite_output_status'].eq('finite').sum())
    selected_id = str(selected.get('candidate_id') or '')
    status = (
        'current_data_binary_review_triage_available'
        if selected_id
        else 'binary_review_triage_not_estimable'
    )
    return {
        'workflow': 'label_free_roi_embedding_atlas',
        'product_direction': 'binary_no_low_vs_moderate_severe_review_triage',
        'overall_status': status,
        'selected_candidate_id': selected_id,
        'selected_target_kind': selected.get('target_kind', ''),
        'selected_feature_family': selected.get('feature_family', ''),
        'threshold': float(selected.get('threshold') or 0.5),
        'threshold_selection_objective': selected.get(
            'threshold_selection_objective', ''
        ),
        'selection_rule': selected.get('selection_rule', ''),
        'primary_target_support': primary,
        'sensitivity_target_support': sensitivity,
        'finite_candidate_count': finite_candidates,
        'candidate_metric_rows': int(len(metrics)),
        'confidence_intervals': intervals,
        'model_support': model_support,
        'anchor_support': {
            'candidate_anchor_cluster_count': int(len(anchor_manifest)),
            'blocked_cluster_count': int(len(blocked_clusters)),
            'recovered_anchor_count': int(len(recovered_anchors)),
        },
        'claim_boundary': BINARY_TRIAGE_CLAIM_BOUNDARY,
        'reviewer_next_step': (
            'Open evidence/binary_triage_review.html for routed case review and '
            'predictions/binary_triage_predictions.csv for all row-level routes.'
        ),
    }


def _render_binary_triage_verdict_md(verdict: dict[str, Any]) -> str:
    return f"""# Binary review triage verdict

Status: `{verdict.get('overall_status')}`

Selected candidate: `{verdict.get('selected_candidate_id') or 'none'}`

Operating threshold: `{verdict.get('threshold')}` selected by
`{verdict.get('threshold_selection_objective')}`.

Primary target: no/low is `score <= 0.5`, moderate/severe is `score >= 1.5`,
and `score == 1.0` is routed as borderline review outside the primary binary
training target.

Sensitivity target: no/low inclusive is `score <= 1.0`, moderate/severe is
`score >= 1.5`.

{BINARY_TRIAGE_CLAIM_BOUNDARY}

## First Read

- `predictions/binary_triage_predictions.csv`
- `evidence/binary_triage_review.html`
- `evidence/binary_triage_explanations.csv`
- `summary/binary_triage_metrics.csv`
- `summary/binary_triage_metric_intervals.json`
"""


def _render_binary_triage_index(verdict: dict[str, Any]) -> str:
    return f"""# Binary no/low versus moderate/severe review triage

Status: `{verdict.get('overall_status')}`

{BINARY_TRIAGE_CLAIM_BOUNDARY}

## Open First

- `summary/binary_triage_verdict.md`
- `evidence/binary_triage_review.html`
- `predictions/binary_triage_predictions.csv`
- `summary/binary_triage_metrics.csv`
- `summary/binary_triage_metric_intervals.json`
- `evidence/binary_triage_explanations.csv`

## Route Meanings

- `likely_no_low_review`: model score is away from the threshold toward no/low.
- `likely_moderate_severe_review`: model score is away from the threshold toward moderate/severe.
- `uncertain_binary_review`: model score is near the operating threshold.
- `borderline_score_review`: original score is exactly 1.0 and is not part of the primary binary training target.
- `blocked_cluster_manual_review`: atlas adjudication blocked the source cluster as mixed, source/batch, or ROI/mask artifact.
"""


def _write_binary_triage_review_html(
    output_dir: Path, predictions: pd.DataFrame, explanations: pd.DataFrame
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _binary_review_rows(predictions, explanations)
    total_rows = int(len(predictions))
    selected_rows = len(rows)
    route_counts = _binary_review_route_counts(predictions)
    route_count_html = ''.join(
        f'<span class="count"><strong>{html.escape(str(count))}</strong> '
        f'{html.escape(label)}</span>'
        for label, count in route_counts
    )
    cards = '\n'.join(_binary_review_card(row) for row in rows)
    payload = html.escape(
        json.dumps(
            _json_ready(
                {
                    'rows': rows,
                    'summary': {
                        'total_predictions': total_rows,
                        'sampled_review_cards': selected_rows,
                        'max_review_cards': BINARY_TRIAGE_REVIEW_MAX_ROWS,
                    },
                }
            ),
            allow_nan=False,
        ),
        quote=False,
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Binary endotheliosis review triage</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1f2933}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #d8dde6;padding:14px 20px;z-index:10}}
h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:15px;margin:0 0 5px}}p{{margin:4px 0;color:#52606d}}main{{padding:18px 20px 40px}}
button{{border:1px solid #9fb3c8;background:#fff;border-radius:6px;padding:7px 10px;cursor:pointer}}button.primary{{background:#1f5eff;color:#fff;border-color:#1f5eff}}
.queue-summary{{border:1px solid #bcccdc;border-left:4px solid #1f5eff;background:#f8fafc;border-radius:6px;padding:9px 10px;margin:10px 0}}
.queue-summary strong{{color:#102a43}}.counts{{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}}.count{{background:#e6f0ff;border:1px solid #bcccdc;border-radius:999px;padding:3px 8px;font-size:12px;color:#334e68}}
.case{{background:white;border:1px solid #d8dde6;border-radius:8px;margin:0 0 16px;overflow:hidden}}
.head{{background:#eef2f7;padding:10px 12px}}.grid{{display:grid;grid-template-columns:1fr 1fr minmax(360px,440px);gap:10px;padding:10px}}
figure{{margin:0}}figcaption{{font-size:12px;color:#52606d;margin-bottom:4px}}img{{width:100%;max-height:360px;object-fit:contain;background:#111827;border-radius:4px}}
.meta{{font-size:13px;color:#52606d;line-height:1.45}}.route{{font-weight:700;color:#1f2933}}.explain{{font-size:12px;color:#334e68;overflow-wrap:anywhere}}
.recommendation{{border:1px solid #bcccdc;background:#f8fafc;border-radius:6px;padding:10px;margin-bottom:10px}}.model-answer{{display:grid;gap:2px;border-bottom:1px solid #d8dde6;padding-bottom:8px;margin-bottom:8px}}.model-answer span{{font-size:12px;color:#52606d;text-transform:uppercase}}.model-answer strong{{font-size:18px;color:#102a43}}
.answer{{font-size:15px;font-weight:700;color:#102a43}}.action{{font-size:13px;color:#334e68;line-height:1.45}}
.details{{border-top:1px solid #d8dde6;margin-top:10px;padding-top:8px}}summary{{cursor:pointer;color:#334e68;font-weight:600}}
.controls{{display:grid;gap:8px;margin-top:10px}}label{{display:grid;gap:3px;font-size:12px;color:#334e68}}select,textarea{{width:100%;box-sizing:border-box;border:1px solid #bcccdc;border-radius:5px;padding:6px;font:inherit;background:white}}textarea{{min-height:72px;resize:vertical}}
@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Binary endotheliosis review triage</h1><p>{html.escape(BINARY_TRIAGE_CLAIM_BOUNDARY)}</p><div class="queue-summary"><strong>Review scope: {selected_rows} sampled QA cases from {total_rows} total predictions.</strong><p>This page is not asking you to review every row. It is a bounded, route-stratified quality-control sample focused on blocked, borderline, uncertain, and representative model calls.</p><p>Stop rule: review the cards shown here, export the CSV beside this HTML file, then use the full prediction table only if this sample exposes a systematic failure pattern.</p><div class="counts">{route_count_html}</div></div><button class="primary" id="downloadCsv" type="button">Save review CSV</button> <button id="clearSaved" type="button">Clear saved values</button> <span id="status">Autosaves in this browser. Static HTML cannot silently write files beside itself; the save button will ask where to put the CSV when the browser allows it.</span></header>
<main>{cards or '<p>No binary triage rows were generated.</p>'}</main>
<script id="data" type="application/json">{payload}</script>
<script>
const data = JSON.parse(document.getElementById('data').textContent);
const storageKey = 'eq.binary_triage_review.' + location.pathname;
const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
function persist(id, key, value) {{
  saved[id] = {{...(saved[id] || {{}}), [key]: value}};
  localStorage.setItem(storageKey, JSON.stringify(saved));
  document.getElementById('status').textContent = 'Saved locally at ' + new Date().toLocaleTimeString();
}}
document.querySelectorAll('select,textarea').forEach(el => {{
  const state = saved[el.dataset.id] || {{}};
  if (state[el.dataset.key]) el.value = state[el.dataset.key];
  el.addEventListener('change', ev => persist(ev.target.dataset.id, ev.target.dataset.key, ev.target.value));
  el.addEventListener('input', ev => persist(ev.target.dataset.id, ev.target.dataset.key, ev.target.value));
}});
function csvText() {{
  const cols = ['review_id','review_sample_scope','review_sample_size','total_prediction_rows','atlas_row_id','subject_id','subject_image_id','score','binary_target_primary','model_answer','model_recommendation','recommended_human_action','predicted_probability_moderate_severe','binary_decision','uncertainty_label','final_review_route','triage_review_decision','review_urgency','reviewer_notes','reviewed_at','roi_image_path','roi_mask_path'];
  const rows = data.rows.map(row => {{
    const id = 'triage-' + row.atlas_row_id;
    const state = saved[id] || {{}};
    return {{...row, ...state, review_id: id, review_sample_scope: 'bounded_route_stratified_qa_sample', review_sample_size: data.summary.sampled_review_cards, total_prediction_rows: data.summary.total_predictions, reviewed_at: new Date().toISOString()}};
  }});
  return [cols.join(',')].concat(rows.map(row => cols.map(col => `"${{String(row[col] ?? '').replaceAll('"','""')}}"`).join(','))).join('\\n');
}}
function downloadCsv(csv) {{
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'binary_triage_review_export.csv';
  link.click();
  URL.revokeObjectURL(link.href);
}}
async function exportCsv() {{
  const csv = csvText();
  if (window.showSaveFilePicker) {{
    try {{
      const handle = await window.showSaveFilePicker({{
        suggestedName: 'binary_triage_review_export.csv',
        types: [{{description: 'CSV review export', accept: {{'text/csv': ['.csv']}}}}],
      }});
      const writable = await handle.createWritable();
      await writable.write(csv);
      await writable.close();
      document.getElementById('status').textContent = 'Review CSV saved. Keep it next to this HTML file.';
      return;
    }} catch (error) {{
      if (error && error.name === 'AbortError') return;
    }}
  }}
  downloadCsv(csv);
  document.getElementById('status').textContent = 'Review CSV downloaded. Move it next to this HTML file before rerunning the workflow.';
}}
document.getElementById('downloadCsv').addEventListener('click', exportCsv);
document.getElementById('clearSaved').addEventListener('click', () => {{ if (confirm('Clear saved review values?')) {{ localStorage.removeItem(storageKey); location.reload(); }} }});
</script>
</body></html>"""
    path = output_dir / 'binary_triage_review.html'
    path.write_text(document, encoding='utf-8')
    return path


def _binary_review_rows(
    predictions: pd.DataFrame,
    explanations: pd.DataFrame,
    *,
    max_rows: int = BINARY_TRIAGE_REVIEW_MAX_ROWS,
) -> list[dict[str, Any]]:
    merged = predictions.merge(explanations, on='atlas_row_id', how='left')
    route_order = {
        'blocked_cluster_manual_review': 0,
        'borderline_score_review': 1,
        'uncertain_binary_review': 2,
        'likely_moderate_severe_review': 3,
        'likely_no_low_review': 4,
    }
    merged['route_order'] = merged['final_review_route'].map(route_order).fillna(9)
    merged['prob_margin'] = (
        pd.to_numeric(merged['predicted_probability_moderate_severe'], errors='coerce')
        - 0.5
    ).abs()
    per_route = max(6, max_rows // max(1, merged['final_review_route'].nunique()))
    selected = (
        merged.sort_values(['route_order', 'prob_margin'], ascending=[True, True])
        .groupby('final_review_route', dropna=False)
        .head(per_route)
        .head(max_rows)
    )
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        payload = {key: row.get(key, '') for key in selected.columns}
        recommendation, action = _binary_review_plain_language(payload)
        payload['model_answer'] = _binary_review_model_answer(payload)
        payload['model_recommendation'] = recommendation
        payload['recommended_human_action'] = action
        rows.append(payload)
    return rows


def _binary_review_route_counts(predictions: pd.DataFrame) -> list[tuple[str, int]]:
    route_order = [
        'blocked_cluster_manual_review',
        'borderline_score_review',
        'uncertain_binary_review',
        'likely_moderate_severe_review',
        'likely_no_low_review',
    ]
    counts = predictions.get('final_review_route', pd.Series(dtype=object)).value_counts(
        dropna=False
    )
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for route in route_order:
        count = int(counts.get(route, 0))
        if count:
            result.append((_binary_review_route_label(route), count))
            seen.add(route)
    for route, count in counts.items():
        route_key = '' if pd.isna(route) else str(route)
        if route_key not in seen:
            result.append((_binary_review_route_label(route_key), int(count)))
    return result


def _binary_review_route_label(route: str) -> str:
    labels = {
        'blocked_cluster_manual_review': 'blocked/manual-review route',
        'borderline_score_review': 'borderline-score route',
        'uncertain_binary_review': 'uncertain-model route',
        'likely_moderate_severe_review': 'likely moderate/severe route',
        'likely_no_low_review': 'likely no/low route',
    }
    return labels.get(route, route or 'missing route')


def _binary_review_model_answer(row: dict[str, Any]) -> str:
    route = str(row.get('final_review_route', ''))
    decision = str(row.get('binary_decision', ''))
    if route == 'blocked_cluster_manual_review':
        return 'No model answer: manual review required'
    if route == 'borderline_score_review':
        return 'No model answer: borderline score review'
    if route == 'uncertain_binary_review':
        return 'Uncertain: human review required'
    if route == 'likely_moderate_severe_review' or decision == 'moderate_severe':
        return 'Moderate/severe'
    if route == 'likely_no_low_review' or decision == 'no_low':
        return 'No/low'
    return decision or 'Not available'


def _binary_review_plain_language(row: dict[str, Any]) -> tuple[str, str]:
    route = str(row.get('final_review_route', ''))
    decision = str(row.get('binary_decision', ''))
    if route == 'blocked_cluster_manual_review':
        return (
            'Model is not trusted for this case because the atlas cluster or ROI/mask evidence is problematic.',
            'Review the ROI and mask first. Exclude it if the crop or mask is not gradeable; otherwise choose the human grade group.',
        )
    if route == 'borderline_score_review':
        return (
            'Original score is 1.0, so this case is intentionally not forced into the primary binary model.',
            'Use the images to decide whether this should remain borderline, be corrected to no/low, or be corrected to moderate/severe.',
        )
    if route == 'likely_moderate_severe_review':
        return (
            'Model thinks this case is likely moderate/severe.',
            'Accept the model only if the ROI image and mask show moderate/severe endotheliosis; otherwise correct it.',
        )
    if route == 'likely_no_low_review':
        return (
            'Model thinks this case is likely no/low.',
            'Accept the model only if the ROI image and mask show no/low endotheliosis; otherwise correct it.',
        )
    if route == 'uncertain_binary_review':
        return (
            'Model is uncertain and is asking for human review.',
            'Choose the human grade group from the images, or request another human review if you cannot decide confidently.',
        )
    return (
        f'Model decision is {decision or "not available"}.',
        'Use the ROI image and mask as the source of truth for the review decision.',
    )


def _binary_review_card(row: dict[str, Any]) -> str:
    probability = pd.to_numeric(
        pd.Series([row.get('predicted_probability_moderate_severe')]),
        errors='coerce',
    ).iloc[0]
    probability_text = (
        'not estimated' if pd.isna(probability) else f'{float(probability) * 100:.0f}%'
    )
    recommendation = str(row.get('model_recommendation', ''))
    action = str(row.get('recommended_human_action', ''))
    model_answer = str(row.get('model_answer', ''))
    route_label = _binary_review_route_label(str(row.get('final_review_route', '')))
    return f"""
<article class="case">
  <div class="head"><strong>Atlas row {html.escape(str(row.get('atlas_row_id', '')))} | score {html.escape(str(row.get('score', '')))}</strong><div class="meta">subject {html.escape(str(row.get('subject_id', '')))} | image {html.escape(str(row.get('subject_image_id', '')))}</div></div>
  <div class="grid">
    <figure><figcaption>ROI image</figcaption><img src="{html.escape(_review_image_src(row.get('roi_image_path', '')))}" alt="ROI image"></figure>
    <figure><figcaption>ROI mask</figcaption><img src="{html.escape(_review_image_src(row.get('roi_mask_path', '')))}" alt="ROI mask"></figure>
    <section>
      <div class="recommendation">
        <h2>Model recommendation</h2>
        <p class="model-answer"><span>Model answer</span><strong>{html.escape(model_answer)}</strong></p>
        <p class="answer">{html.escape(recommendation)}</p>
        <p class="action">{html.escape(action)}</p>
      </div>
      <p class="route">Why this case is shown: {html.escape(route_label)}</p>
      <p class="meta">This card is part of the bounded QA sample, not a request to review every prediction.</p>
      <p class="meta">Model probability of moderate/severe: {html.escape(probability_text)} | model class: {html.escape(str(row.get('binary_decision', '')))} | confidence: {html.escape(str(row.get('uncertainty_label', '')))}</p>
      <div class="controls">
        <label>Your decision<select data-id="triage-{html.escape(str(row.get('atlas_row_id', '')))}" data-key="triage_review_decision"><option value="">choose</option><option value="accept_model_recommendation">accept model recommendation</option><option value="human_no_low">human says no/low</option><option value="human_moderate_severe">human says moderate/severe</option><option value="needs_second_human_review">needs second human review</option><option value="exclude_bad_roi_or_mask">exclude: bad ROI or mask</option></select></label>
        <label>Follow-up status<select data-id="triage-{html.escape(str(row.get('atlas_row_id', '')))}" data-key="review_urgency"><option value="">choose</option><option value="routine">routine</option><option value="urgent_possible_model_error">urgent: possible model error</option><option value="defer_not_useful_now">defer: not useful now</option></select></label>
        <label>reviewer notes<textarea data-id="triage-{html.escape(str(row.get('atlas_row_id', '')))}" data-key="reviewer_notes"></textarea></label>
      </div>
      <details class="details"><summary>Model diagnostics</summary>
        <p class="meta">Primary target: {html.escape(str(row.get('binary_target_primary', '')))} | sensitivity target: {html.escape(str(row.get('binary_target_sensitivity', '')))}</p>
        <p class="meta">Nearest anchor: row {html.escape(str(row.get('nearest_anchor_atlas_row_id', '')))} ({html.escape(str(row.get('nearest_anchor_class', '')))}) distance {html.escape(str(row.get('nearest_anchor_distance', '')))}</p>
        <p class="meta">Warnings: {html.escape(str(row.get('source_cohort_warning', row.get('source_warning', ''))))}</p>
        <p class="explain"><strong>Top feature evidence:</strong> {html.escape(str(row.get('top_feature_contributions', '')))}</p>
        <p class="explain"><strong>Feature-family evidence:</strong> {html.escape(str(row.get('feature_family_contributions', '')))}</p>
        <p class="explain">{html.escape(str(row.get('explanation_claim_boundary', '')))}</p>
      </details>
    </section>
  </div>
</article>"""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, (str, bytes, Path)):
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
    return value


def _build_verdict(
    *,
    status: str,
    feature_spaces: list[str],
    methods: list[str],
    blockers: list[str],
    warnings: list[str],
    review_queue_count: int,
    selected_atlas_view: dict[str, str] | None,
    adjudication_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adjudication_summary = adjudication_summary or {}
    adjudication_status = str(adjudication_summary.get('status', 'not_provided'))
    next_action = (
        'Inspect INDEX.md, evidence/embedding_atlas_review.html, '
        'evidence/atlas_final_adjudication_outcome.md, and '
        'binary_review_triage/INDEX.md.'
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
        'adjudication_status': adjudication_status,
        'adjudication_summary': adjudication_summary,
        'score_change_count': int(
            adjudication_summary.get('score_correction_count', 0) or 0
        ),
        'recovered_anchor_count': int(
            adjudication_summary.get('recovered_anchor_count', 0) or 0
        ),
        'candidate_anchor_cluster_count': int(
            adjudication_summary.get('candidate_anchor_cluster_count', 0) or 0
        ),
        'blocked_cluster_count': int(
            adjudication_summary.get('blocked_cluster_count', 0) or 0
        ),
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
        'adjudication_status': 'explicit_in_summary_atlas_verdict_json',
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

## Adjudication And Binary Triage

- Adjudication status: `{verdict.get('adjudication_status', 'not_provided')}`
- Score changes: `{verdict.get('score_change_count', 0)}`
- Recovered anchors: `{verdict.get('recovered_anchor_count', 0)}`
- Candidate anchor clusters: `{verdict.get('candidate_anchor_cluster_count', 0)}`
- Blocked clusters: `{verdict.get('blocked_cluster_count', 0)}`
- Final adjudication outcome: `../evidence/atlas_final_adjudication_outcome.md`
- Binary triage index: `../binary_review_triage/INDEX.md`

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
- `evidence/atlas_final_adjudication_outcome.md`
- `evidence/atlas_score_corrections.csv`
- `evidence/atlas_recovered_anchor_examples.csv`
- `evidence/atlas_adjudicated_anchor_manifest.csv`
- `evidence/atlas_blocked_cluster_manifest.csv`
- `binary_review_triage/INDEX.md`
- `binary_review_triage/evidence/binary_triage_review.html`
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
