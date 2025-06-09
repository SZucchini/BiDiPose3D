"""Visualization of 2D pose animation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from bidipose.statics.bone import h36m_bones


def set_lines(x: np.ndarray, y: np.ndarray, bones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set lines for 2D visualization.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.
        bones (np.ndarray): Bone connections.

    Returns:
        tuple[np.ndarray, np.ndarray]: Line coordinates for X, Y.

    """
    line_x, line_y = [], []
    for bone in bones:
        line_x.append([x[bone[0]], x[bone[1]]])
        line_y.append([y[bone[0]], y[bone[1]]])
    return np.array(line_x), np.array(line_y)


def vis_pose2d(
    pred_pose: np.ndarray,
    gt_pose: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "2D Pose Visualization",
) -> animation.FuncAnimation:
    """Visualize 2D poses from two views.

    Args:
        pred_pose (np.ndarray): Predicted 2D poses from two-views (T, J, 6).
            First 2 channels are for view1, channels 4-5 are for view2.
        gt_pose (np.ndarray | None): Ground truth 2D poses from two-views (T, J, 6).
            First 2 channels are for view1, channels 4-5 are for view2.
        save_path (str | None): Path to save the animation. If None, the animation is returned.
        title (str): Title of the plot.

    Returns:
        animation.FuncAnimation: Animation object for the 2D pose visualization.

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    pred_view1 = pred_pose[:, :, :2]
    pred_view2 = pred_pose[:, :, 3:5]
    xmin = min(np.min(pred_view1[:, :, 0]), np.min(pred_view2[:, :, 0])) - 0.15
    xmax = max(np.max(pred_view1[:, :, 0]), np.max(pred_view2[:, :, 0])) + 0.15
    ymin = min(np.min(pred_view1[:, :, 1]), np.min(pred_view2[:, :, 1])) - 0.15
    ymax = max(np.max(pred_view1[:, :, 1]), np.max(pred_view2[:, :, 1])) + 0.15
    if gt_pose is not None:
        gt_view1 = gt_pose[:, :, :2]
        gt_view2 = gt_pose[:, :, 3:5]
        xmin = min(xmin, np.min(gt_view1[:, :, 0]), np.min(gt_view2[:, :, 0])) - 0.15
        xmax = max(xmax, np.max(gt_view1[:, :, 0]), np.max(gt_view2[:, :, 0])) + 0.15
        ymin = min(ymin, np.min(gt_view1[:, :, 1]), np.min(gt_view2[:, :, 1])) - 0.15
        ymax = max(ymax, np.max(gt_view1[:, :, 1]), np.max(gt_view2[:, :, 1])) + 0.15

    def draw_skeleton(
        ax: plt.Axes,
        pred: np.ndarray,
        gt: np.ndarray | None = None,
        x_min: float = -1,
        x_max: float = 1,
        y_min: float = -1,
        y_max: float = 1,
    ) -> None:
        """Draw skeleton."""
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.invert_xaxis()
        ax.invert_yaxis()

        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 1], "k.", label="GT", markersize=5, alpha=0.5)
            x_bone_gt, y_bone_gt = set_lines(gt[:, 0], gt[:, 1], h36m_bones)
            for x, y in zip(x_bone_gt, y_bone_gt, strict=False):
                ax.plot(x, y, "k-", linewidth=2, alpha=0.5)

        ax.plot(pred[:, 0], pred[:, 1], "r.", label="Pred", markersize=5, alpha=0.8)
        x_bone_pred, y_bone_pred = set_lines(pred[:, 0], pred[:, 1], h36m_bones)
        for x, y in zip(x_bone_pred, y_bone_pred, strict=False):
            ax.plot(x, y, "r-", linewidth=2, alpha=0.8)

        ax.legend()

    def update_frame(fc: int) -> None:
        """Update frame."""
        if gt_pose is not None:
            draw_skeleton(ax1, pred_view1[fc], gt_view1[fc], xmin, xmax, ymin, ymax)
        else:
            draw_skeleton(ax1, pred_view1[fc], None, xmin, xmax, ymin, ymax)
        ax1.set_title("View 1")

        if gt_pose is not None:
            draw_skeleton(ax2, pred_view2[fc], gt_view2[fc], xmin, xmax, ymin, ymax)
        else:
            draw_skeleton(ax2, pred_view2[fc], None, xmin, xmax, ymin, ymax)
        ax2.set_title("View 2")

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=pred_pose.shape[0],
        interval=30,
        repeat=False,
    )
    if save_path is not None:
        anim.save(save_path, writer="ffmpeg", fps=30)
        plt.close(fig)
        del anim
    else:
        return anim
