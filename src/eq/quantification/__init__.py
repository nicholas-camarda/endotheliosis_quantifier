"""Quantification workflows for contract enforcement, ROI extraction, and scoring."""

from .cohorts import (
    build_current_accessible_cohorts,
    build_dox_mask_quality_audit,
    canonical_manifest_columns,
    enrich_unified_manifest,
    validate_unified_manifest,
)
from .dataset import build_scored_example_table, load_standardized_metadata
from .embeddings import extract_encoder_embeddings_from_rois
from .migration import (
    generate_mapping_template,
    inventory_raw_project,
    migrate_raw_project_to_canonical,
)
from .ordinal import CanonicalOrdinalClassifier, run_grouped_ordinal_experiment
from .pipeline import (
    prepare_quantification_contract,
    run_contract_first_quantification,
    run_endotheliosis_scoring_pipeline,
)
from .roi import extract_rois_for_scored_examples

__all__ = [
    'inventory_raw_project',
    'generate_mapping_template',
    'migrate_raw_project_to_canonical',
    'load_standardized_metadata',
    'build_scored_example_table',
    'build_current_accessible_cohorts',
    'build_dox_mask_quality_audit',
    'canonical_manifest_columns',
    'enrich_unified_manifest',
    'validate_unified_manifest',
    'extract_rois_for_scored_examples',
    'extract_encoder_embeddings_from_rois',
    'CanonicalOrdinalClassifier',
    'run_grouped_ordinal_experiment',
    'prepare_quantification_contract',
    'run_contract_first_quantification',
    'run_endotheliosis_scoring_pipeline',
]
