import pytest

from eq.training.losses import TverskyLoss, loss_metadata, make_loss


def test_empty_loss_uses_default_fastai_loss_metadata():
    metadata = loss_metadata("")

    assert metadata["requested_loss_name"] is None
    assert metadata["resolved_loss_class"] is None
    assert metadata["loss_parameters"] == {}
    assert metadata["false_positive_penalizing"] is False
    assert make_loss("") is None


def test_tversky_fp_loss_is_explicitly_false_positive_penalizing():
    metadata = loss_metadata("tversky_fp")
    loss = make_loss("tversky_fp")

    assert isinstance(loss, TverskyLoss)
    assert metadata["resolved_loss_class"] == "TverskyLoss"
    assert metadata["loss_parameters"] == {"alpha": 0.7, "beta": 0.3, "smooth": 1.0}
    assert metadata["false_positive_penalizing"] is True
    assert loss.alpha == pytest.approx(0.7)
    assert loss.beta == pytest.approx(0.3)


def test_equal_tversky_is_not_reported_as_false_positive_penalizing():
    metadata = loss_metadata("tversky")

    assert metadata["loss_parameters"]["alpha"] == 0.5
    assert metadata["loss_parameters"]["beta"] == 0.5
    assert metadata["false_positive_penalizing"] is False


def test_tversky_parameters_are_recorded_exactly():
    metadata = loss_metadata("tversky:alpha=0.8,beta=0.2,smooth=2")
    loss = make_loss("tversky:alpha=0.8,beta=0.2,smooth=2")

    assert metadata["loss_parameters"] == {"alpha": 0.8, "beta": 0.2, "smooth": 2.0}
    assert metadata["false_positive_penalizing"] is True
    assert loss.alpha == pytest.approx(0.8)
    assert loss.beta == pytest.approx(0.2)
    assert loss.smooth == pytest.approx(2.0)


def test_unsupported_loss_fails_before_training():
    with pytest.raises(ValueError, match="Unsupported segmentation loss"):
        make_loss("not_a_loss")


def test_invalid_tversky_parameters_fail_before_training():
    with pytest.raises(ValueError, match="Tversky loss requires"):
        make_loss("tversky:alpha=-1,beta=0.5")
