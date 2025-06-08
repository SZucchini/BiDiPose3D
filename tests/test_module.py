from unittest.mock import MagicMock, patch

import pytest
import torch

from bidipose.diffusion.module import DiffusionLightningModule


class DummyModel(torch.nn.Module):
    def forward(self, x, quat, trans, t):
        return x + 1, quat + 1, trans + 1

    def parameters(self):
        return [torch.nn.Parameter(torch.randn(1))]


class DummySampler:
    timesteps = 10

    def q_sample(self, x, quat, trans, t):
        return x + 0.1, quat + 0.1, trans + 0.1

    def sample(self, model, x_shape, quat_shape, trans_shape, **kwargs):
        x = torch.zeros(x_shape)
        quat = torch.zeros(quat_shape)
        trans = torch.zeros(trans_shape)
        return x, quat, trans


@pytest.fixture
def module():
    model = DummyModel()
    sampler = DummySampler()
    optimizer_params = {"lr": 1e-3}
    return DiffusionLightningModule(
        model=model,
        sampler=sampler,
        optimizer_name="Adam",
        optimizer_params=optimizer_params,
        num_validation_batches_to_sample=2,
        num_validation_batches_to_inpaint=2,
        num_plot_sample=1,
        num_plot_inpaint=1,
        inpainting_spatial_name=None,
        inpainting_temporal_interval=None,
        inpainting_camera_index=None,
    )


def test_forward(module):
    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    t = torch.randint(0, 10, (2,))
    out = module.forward(x, quat, trans, t)
    assert isinstance(out, tuple)
    assert all(isinstance(o, torch.Tensor) for o in out)
    assert out[0].shape == x.shape


def test_sample(module):
    x_shape = (2, 81, 17, 6)
    quat_shape = (2, 4)
    trans_shape = (2, 3)
    x, quat, trans = module.sample(x_shape, quat_shape, trans_shape)
    assert x.shape == x_shape
    assert quat.shape == quat_shape
    assert trans.shape == trans_shape


def test_inpaint(module):
    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    x_out, quat_out, trans_out = module.inpaint(x, quat, trans, x_mask=mask, quat_mask=None, trans_mask=None)
    assert x_out.shape == x.shape
    assert quat_out.shape == quat.shape
    assert trans_out.shape == trans.shape


def test_training_step_logs(module):
    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    batch = (x, quat, trans)
    module.log = MagicMock()
    loss = module.training_step(batch, 0)
    assert module.log.call_count == 1


def test_validation_step_logs_and_batches(module):
    x = torch.randn(2, 81, 17, 6)
    quat = torch.randn(2, 4)
    trans = torch.randn(2, 3)
    batch = (x, quat, trans)
    module.log = MagicMock()
    module.validation_batches = []
    module.num_validation_batches_to_inpaint = 2
    module.validation_step(batch, 0)
    assert module.log.call_count == 1
    assert len(module.validation_batches) == 1


def test_configure_optimizers(module):
    optimizer = module.configure_optimizers()
    assert hasattr(optimizer, "step")
    assert hasattr(optimizer, "zero_grad")


def test_process_data_for_logging(module):
    x = [torch.randn(2, 81, 17, 6), torch.randn(2, 81, 17, 6)]
    x_gt = [torch.randn(2, 81, 17, 6), torch.randn(2, 81, 17, 6)]
    arr_x, arr_x_gt = module._process_data_for_logging(x, x_gt)
    assert isinstance(arr_x, (list, torch.Tensor, type(arr_x_gt)))
    assert arr_x.shape == arr_x_gt.shape


@patch("bidipose.diffusion.module.vis_pose2d")
@patch("bidipose.diffusion.module.vis_pose3d")
def test_log_animation(mock_vis_pose3d, mock_vis_pose2d, module, tmp_path):
    # Setup dummy animations
    class DummyAni:
        def save(self, path, writer, fps):
            pass

    mock_vis_pose2d.return_value = DummyAni()
    mock_vis_pose3d.return_value = DummyAni()
    module.trainer = MagicMock()
    module.trainer.default_root_dir = str(tmp_path)
    from unittest.mock import PropertyMock

    type(module).current_epoch = PropertyMock(return_value=1)
    # Patch the logger property since it has no setter
    mock_logger = MagicMock()
    mock_logger.log_video = MagicMock()
    with patch.object(type(module), "logger", new_callable=PropertyMock, return_value=mock_logger):
        x = [torch.randn(2, 81, 17, 6)]
        quat = [torch.randn(2, 4)]
        trans = [torch.randn(2, 3)]
        x_gt = [torch.randn(2, 81, 17, 6)]
        quat_gt = [torch.randn(2, 4)]
        trans_gt = [torch.randn(2, 3)]
        module._log_animation(
            key1="test", key2="test", x=x, quat=quat, trans=trans, x_gt=x_gt, quat_gt=quat_gt, trans_gt=trans_gt
        )
        assert mock_logger.log_video.call_count == 2
