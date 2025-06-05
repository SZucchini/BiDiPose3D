import torch


def get_spatial_mask(x: torch.Tensor, spatial_index: torch.Tensor) -> torch.Tensor:
    """
    Get spatial mask for a given tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, J, 2*2).
        spatial_index (torch.Tensor): Index tensor to set the mask to 1 along the third, spatial dimension

    Returns:
        torch.Tensor: Spatial mask of shape (B, T, J, 2*2).
    """
    mask = torch.zeros((x.size(2),), device=x.device, dtype=torch.bool)
    mask[spatial_index] = 1
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(x)
    return mask

def get_temporal_mask(x: torch.Tensor, temporal_index: torch.Tensor) -> torch.Tensor:
    """
    Get temporal mask for a given tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, J, 2*2).
        temporal_index (torch.Tensor): Index tensor to set the mask to 1 along the second, temporal dimension.

    Returns:
        torch.Tensor: Temporal mask of shape (B, T, J, 2*2).
    """
    mask = torch.zeros((x.size(1),), device=x.device, dtype=torch.bool)
    mask[temporal_index] = 1
    mask = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
    return mask

def get_camera_mask(x: torch.Tensor, camera_index: int) -> torch.Tensor:
    """
    Get camera mask for a given tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, J, 2*2).
        camera_index (0 or 1, optional): Index of the camera to set the mask to 1.

    Returns:
        torch.Tensor: Camera mask of shape (B, T, J, 2*2).
    """
    mask = torch.zeros((2,2), device=x.device, dtype=torch.bool)
    mask[camera_index] = 1
    mask = mask.flatten().unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(x)
    return mask
