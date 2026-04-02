"""Quantification workflows for contract enforcement, ROI extraction, and scoring."""

from .migration import (
    inventory_raw_project,
    generate_mapping_template,
    migrate_raw_project_to_canonical,
)
from .dataset import build_scored_example_table, load_standardized_metadata
from .roi import extract_rois_for_scored_examples
from .embeddings import extract_encoder_embeddings_from_rois
from .ordinal import run_grouped_ordinal_experiment
from .pipeline import (
    prepare_quantification_contract,
    run_contract_first_quantification,
    run_endotheliosis_scoring_pipeline,
)

__all__ = [
    'inventory_raw_project',
    'generate_mapping_template',
    'migrate_raw_project_to_canonical',
    'load_standardized_metadata',
    'build_scored_example_table',
    'extract_rois_for_scored_examples',
    'extract_encoder_embeddings_from_rois',
    'run_grouped_ordinal_experiment',
    'prepare_quantification_contract',
    'run_contract_first_quantification',
    'run_endotheliosis_scoring_pipeline',
]
