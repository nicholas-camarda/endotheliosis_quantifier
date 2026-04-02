"""Embedding extraction from the frozen glomeruli segmentation encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from eq.core.constants import DEFAULT_IMAGE_SIZE
from eq.data_management.model_loading import load_model_safely
from eq.inference.prediction_core import PredictionCore
from eq.training.transfer_learning import _get_encoder_module


def _resolve_encoder_output(features: object) -> torch.Tensor:
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if not isinstance(features, torch.Tensor):
        raise TypeError(f'Encoder output must be a tensor, got {type(features)}')
    return features


def extract_encoder_embeddings_from_rois(
    roi_examples_path: Path,
    segmentation_model_path: Path,
    output_dir: Path,
    expected_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Path]:
    """Extract pooled encoder embeddings from ROI crops."""
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_examples = pd.read_csv(roi_examples_path)
    valid_examples = roi_examples[roi_examples['roi_status'].isin(['matched_component_rank', 'heuristic_component_rank'])].copy()
    if valid_examples.empty:
        raise ValueError('No ROI rows are available for embedding extraction.')

    learner = load_model_safely(str(segmentation_model_path), model_type='glomeruli')
    encoder = _get_encoder_module(learner.model)
    if encoder is None:
        raise RuntimeError('Could not resolve the encoder module from the segmentation model.')
    learner.model.eval()
    encoder.eval()
    device = next(learner.model.parameters()).device
    core = PredictionCore(expected_size=expected_size)

    embeddings: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for embedding_index, row in enumerate(valid_examples.to_dict(orient='records')):
            image = Image.open(Path(str(row['roi_image_path']))).convert('RGB')
            tensor = core.preprocess_image(image).to(device)
            features = _resolve_encoder_output(encoder(tensor))
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1).cpu().numpy()[0]
            embeddings.append(pooled.astype(np.float32))
            row['embedding_index'] = embedding_index
            row['embedding_status'] = 'extracted'
            metadata_rows.append(row)

    embedding_matrix = np.vstack(embeddings).astype(np.float32)
    metadata = pd.DataFrame.from_records(metadata_rows)

    embeddings_path = output_dir / 'encoder_embeddings.npy'
    np.save(embeddings_path, embedding_matrix)
    metadata_path = output_dir / 'embedding_metadata.csv'
    metadata.to_csv(metadata_path, index=False)
    summary = {
        'num_embeddings': int(embedding_matrix.shape[0]),
        'embedding_dim': int(embedding_matrix.shape[1]),
        'model_path': str(segmentation_model_path),
        'pooling': 'adaptive_avg_pool2d_1x1',
        'preprocessing': f'PredictionCore(expected_size={expected_size})',
    }
    summary_path = output_dir / 'embedding_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        'embeddings': embeddings_path,
        'embedding_metadata': metadata_path,
        'embedding_summary': summary_path,
    }
