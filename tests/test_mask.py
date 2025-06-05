import torch

from bidipose.diffusion.utils import get_spatial_mask, get_temporal_mask, get_camera_mask

def test_get_spatial_mask():
    """Test the get_spatial_mask function."""
    x = torch.randn(1, 81, 17, 4)
    spatial_index = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16])
    residual_index = torch.tensor([i for i in range(17) if i not in spatial_index])
    mask = get_spatial_mask(x, spatial_index)
    assert mask.shape == x.shape, "Spatial mask shape mismatch"
    assert torch.all(mask[:, :, spatial_index, :] == 1), "Spatial mask values mismatch"
    assert torch.all(mask[:, :, residual_index, :] == 0), "Spatial mask values mismatch for non-indexed joints"

def test_get_temporal_mask():
    """Test the get_temporal_mask function."""
    x = torch.randn(1, 81, 17, 4)
    temporal_index = torch.tensor([0, 20, 40, 60, 80])
    residual_index = torch.tensor([i for i in range(81) if i not in temporal_index])
    mask = get_temporal_mask(x, temporal_index)
    assert mask.shape == x.shape, "Temporal mask shape mismatch"
    assert torch.all(mask[:, temporal_index, :, :] == 1), "Temporal mask values mismatch"
    assert torch.all(mask[:, residual_index, :, :] == 0), "Temporal mask values mismatch for non-indexed frames"

def test_get_camera_mask():
    """Test the get_camera_mask function."""
    x = torch.randn(1, 81, 17, 4)
    camera_index = 0
    mask = get_camera_mask(x, camera_index)
    assert mask.shape == x.shape, "Camera mask shape mismatch"
    assert torch.all(mask[:, :, :, :2] == 1), "Camera mask values mismatch"
    assert torch.all(mask[:, :, :, 2:] == 0), "Camera mask values mismatch for non-indexed camera"

    camera_index = 1
    mask = get_camera_mask(x, camera_index)
    assert mask.shape == x.shape, "Camera mask shape mismatch for second camera"
    assert torch.all(mask[:, :, :, 2:] == 1), "Camera mask values mismatch for second camera"
    assert torch.all(mask[:, :, :, :2] == 0), "Camera mask values mismatch for non-indexed camera"
