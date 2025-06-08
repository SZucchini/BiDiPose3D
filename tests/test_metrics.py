import torch
from bidipose.eval import metrics

def test_epipolar_error_shape_and_value():
    # (B, T, J, 6)
    x = torch.randn(2, 3, 4, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    mask = torch.ones(3, 4, dtype=torch.bool)
    result = metrics.epipolar_error(x, quat, trans, mask)
    assert result.shape == (), "Output should be a scalar"
    assert torch.is_tensor(result)
    assert result >= 0

def test_epipolar_error_without_mask():
    x = torch.randn(1, 2, 2, 6)
    quat = torch.randn(1, 4)
    trans = torch.randn(1, 3)
    result = metrics.epipolar_error(x, quat, trans)
    assert result.shape == ()
    assert result >= 0

def test_key_point_error_2d_basic():
    x = torch.zeros(1, 2, 2, 6)
    y = torch.ones(1, 2, 2, 6)
    result = metrics.key_point_error_2d(x, y)
    assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

def test_key_point_error_2d_with_mask():
    x = torch.zeros(1, 2, 2, 6)
    y = torch.ones(1, 2, 2, 6)
    mask = torch.ones(1, 2, 2, 6, dtype=torch.bool)
    result = metrics.key_point_error_2d(x, y, mask)
    assert torch.isclose(result, torch.tensor(1.0), atol=1e-5)

def test_key_point_error_2d_with_mask_partial():
    x = torch.zeros(1, 2, 2, 6)
    y = torch.ones(1, 2, 2, 6)
    mask = torch.tensor([[[[True, False, True, False, True, False],
                           [True, False, True, False, True, False]]]])
    x = x.masked_fill(mask, 1.0)
    result = metrics.key_point_error_2d(x, y, mask)
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

def test_camera_direction_error_zero():
    trans_pred = torch.tensor([[1.0, 0.0, 0.0]])
    trans_gt = torch.tensor([[1.0, 0.0, 0.0]])
    result = metrics.camera_direction_error(trans_pred, trans_gt)
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-6), result

def test_camera_direction_error_orthogonal():
    trans_pred = torch.tensor([[1.0, 0.0, 0.0]])
    trans_gt = torch.tensor([[0.0, 1.0, 0.0]])
    result = metrics.camera_direction_error(trans_pred, trans_gt)
    assert torch.isclose(result, torch.tensor(torch.pi/2), atol=1e-5), result

def test_quaternion_conjugate():
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = metrics._quaternion_conjugate(q)
    expected = torch.tensor([[1.0, -2.0, -3.0, -4.0]])
    assert torch.allclose(result, expected)

def test_quaternion_multiply_identity():
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    result = metrics._quaternion_multiply(q1, q2)
    expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(result, expected)

def test_camera_rotation_error_zero():
    quat_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    quat_gt = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    result = metrics.camera_rotation_error(quat_pred, quat_gt)
    assert torch.isclose(result, torch.tensor(0.0), atol=1e-6), result

def test_camera_rotation_error_180_deg():
    quat_pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    quat_gt = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    result = metrics.camera_rotation_error(quat_pred, quat_gt)
    assert torch.isclose(result, torch.tensor(torch.pi), atol=1e-5), result
