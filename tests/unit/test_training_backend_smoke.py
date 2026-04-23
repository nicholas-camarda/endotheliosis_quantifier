import os

import torch


def _select_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _synchronize_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def test_synthetic_segmentation_optimization_step_on_selected_backend():
    device = _select_training_device()
    expected_backend = os.environ.get('EQ_EXPECT_TRAINING_BACKEND')
    if expected_backend:
        assert device.type == expected_backend

    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 1, kernel_size=1),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    images = torch.randn(2, 3, 32, 32, device=device)
    masks = (torch.rand(2, 1, 32, 32, device=device) > 0.5).float()
    before_step = [parameter.detach().cpu().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, masks)
    loss.backward()

    grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    _synchronize_device(device)

    after_step = [parameter.detach().cpu() for parameter in model.parameters()]
    changed = [
        not torch.allclose(before, after)
        for before, after in zip(before_step, after_step)
    ]

    print(f'selected_training_backend={device.type}')
    assert logits.device.type == device.type
    assert masks.device.type == device.type
    assert torch.isfinite(loss.detach().cpu())
    assert grad_norm > 0
    assert any(changed)
