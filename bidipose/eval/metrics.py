import torch

from bidipose.preprocess.utils import essential_from_quat_and_trans_batch


def epipolar_error(
    x: torch.Tensor, quat: torch.Tensor, trans: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the epipolar error between predicted and target poses.

    Args:
        x (torch.Tensor): Input 2D poses from two-views (B, T, J, 3*2).
        quat (torch.Tensor): Quaternions for cam1 to cam2 (B, 4).
        trans (torch.Tensor): Translations for cam1 to cam2 (B, 3).
        mask (torch.Tensor, optional): Mask to apply to the input data (T,J).

    Returns:
        torch.Tensor: Epipolar error.
    """
    essential_mat = essential_from_quat_and_trans_batch(
        quat.detach().cpu().numpy(), trans.detach().cpu().numpy()
    )
    essential_mat = torch.tensor(essential_mat, device=x.device, dtype=x.dtype)
    cam0 = x[:, :, :, :3]
    cam1 = x[:, :, :, 3:]
    epipolar_error = torch.einsum('btjm,btjn,bmn->btj', cam1, cam0, essential_mat)
    if mask is not None:
        epipolar_error = epipolar_error.masked_select(mask)
    return epipolar_error.abs().mean()


def key_point_error_2d(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the 2D keypoint error between two sets of 2D poses.

    Args:
        x (torch.Tensor): First set of 2D poses (B, T, J, 3*2).
        y (torch.Tensor): Second set of 2D poses (B, T, J, 3*2).
        mask (torch.Tensor, optional): Mask to apply to the input data (B, T, J, 3*2).

    Returns:
        torch.Tensor: Mean squared error between the two sets of poses.
    """
    error = torch.square(x - y)
    if mask is not None:
        error = error.masked_select(mask)
    return error.sqrt().mean()


def camera_direction_error(trans_pred: torch.Tensor, trans_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the camera direction error between predicted and ground truth translations.
    Args:
        trans_pred (torch.Tensor): Predicted translations (B, 3).
        trans_gt (torch.Tensor): Ground truth translations (B, 3).
    Returns:
        torch.Tensor: Mean squared error of the camera direction.
    """
    similarity = torch.nn.functional.cosine_similarity(trans_pred, trans_gt, dim=1)
    error_angle = torch.acos(similarity.clamp(-1.0, 1.0))  # Clamp to avoid NaN
    return error_angle.abs().mean()

def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the conjugate of a quaternion.
    Args:
        q (torch.Tensor): Quaternion tensor of shape (B, 4).
    Returns:
        torch.Tensor: Conjugate quaternion tensor of shape (B, 4).
    """
    coef = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    q = q * coef
    return q

def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Args:
        q1 (torch.Tensor): First quaternion (B, 4).
        q2 (torch.Tensor): Second quaternion (B, 4).
    Returns:
        torch.Tensor: Resulting quaternion (B, 4).
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack((w, x, y, z), dim=1)


def camera_rotation_error(quat_pred: torch.Tensor, quat_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the camera rotation error between predicted and ground truth quaternions.
    
    Args:
        quat_pred (torch.Tensor): Predicted quaternions (B, 4).
        quat_gt (torch.Tensor): Ground truth quaternions (B, 4).
    
    Returns:
        torch.Tensor: Mean squared error of the camera rotation.
    """
    quat_pred_inv = _quaternion_conjugate(quat_pred)
    quat_rel = _quaternion_multiply(quat_pred_inv, quat_gt)
    quat_rel = torch.nn.functional.normalize(quat_rel, dim=1)
    error_angle = quat_rel[:,0].clamp(-1, 1).arccos().mul(2).add(torch.pi).remainder(2*torch.pi).sub(torch.pi)
    return error_angle.abs().mean()
