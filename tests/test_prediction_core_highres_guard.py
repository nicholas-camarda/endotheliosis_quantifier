import numpy as np
import pytest
import torch
from PIL import Image

from eq.inference.prediction_core import PredictionCore


class TinySegmentationModel(torch.nn.Module):
    def forward(self, x):
        batch, _channels, height, width = x.shape
        return torch.zeros((batch, 2, height, width), dtype=x.dtype, device=x.device)


def test_prediction_core_rejects_unlabeled_highres_direct_resize():
    core = PredictionCore(expected_size=256)
    image = Image.fromarray(np.zeros((2048, 2448, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="high-resolution full-field"):
        core.predict_segmentation_probability(TinySegmentationModel(), image)


def test_prediction_core_allows_labeled_tile_resize():
    core = PredictionCore(expected_size=256)
    image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))

    probability, audit = core.predict_segmentation_probability(
        TinySegmentationModel(),
        image,
        input_role="tile",
    )

    assert probability.shape == (256, 256)
    assert audit["input_role"] == "tile"
