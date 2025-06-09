"""Visualization of 3D pose animation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import art3d

from bidipose.preprocess.utils import triangulate_points
from bidipose.statics.bone import h36m_bones


def set_lines(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, bones: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set lines for 3D visualization.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.
        z (np.ndarray): Z coordinates.
        bones (np.ndarray): Bone connections.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Line coordinates for X, Y, Z.

    """
    line_x, line_y, line_z = [], [], []
    for bone in bones:
        line_x.append([x[bone[0]], x[bone[1]]])
        line_y.append([y[bone[0]], y[bone[1]]])
        line_z.append([z[bone[0]], z[bone[1]]])
    return np.array(line_x), np.array(line_y), np.array(line_z)


def vis_pose3d(
    pred_pose: np.ndarray,
    pred_quat: np.ndarray,
    pred_trans: np.ndarray,
    gt_pose: np.ndarray | None = None,
    gt_quat: np.ndarray | None = None,
    gt_trans: np.ndarray | None = None,
    save_path: str | None = None,
    title: str = "3D Pose Visualization",
):
    """Visualize 3D pose tringulation results.

    Args:
        pred_pose (np.ndarray): Predicted 2D poses from two-views (T, J, 3*2).
        pred_quat (np.ndarray): Predicted quaternions for cam1 to cam2 (4,).
        pred_trans (np.ndarray): Predicted translations for cam1 to cam2 (3,).
        gt_pose (np.ndarray | None): Ground truth 2D poses from two-views (T, J, 3*2).
        gt_quat (np.ndarray | None): Ground truth quaternions for cam1 to cam2 (4,).
        gt_trans (np.ndarray | None): Ground truth translations for cam1 to cam2 (3,).
        save_path (str | None): Path to save the animation. If None, the animation is returned.
        title (str): Title of the plot.

    Returns:
        animation.FuncAnimation: Animation object for the 3D pose visualization.

    """
    gt_pose3d = None
    pred_pose3d = triangulate_points(pred_pose[:, :, :3], pred_pose[:, :, 3:], pred_quat, pred_trans)
    pred_pose3d[:, :, 2] *= -1
    if gt_pose is not None and gt_quat is not None and gt_trans is not None:
        gt_pose3d = triangulate_points(gt_pose[:, :, :3], gt_pose[:, :, 3:], gt_quat, gt_trans)
        gt_pose3d[:, :, 2] *= -1

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111, projection="3d")

    if gt_pose3d is not None:
        xmin = np.min(gt_pose3d[:, :, 0]) - 0.1
        xmax = np.max(gt_pose3d[:, :, 0]) + 0.1
        ymin = np.min(gt_pose3d[:, :, 1]) - 0.1
        ymax = np.max(gt_pose3d[:, :, 1]) + 0.1
        zmin = np.min(gt_pose3d[:, :, 2]) - 0.1
        zmax = np.max(gt_pose3d[:, :, 2]) + 0.1
    else:
        xmin = np.min(pred_pose3d[:, :, 0]) - 0.1
        xmax = np.max(pred_pose3d[:, :, 0]) + 0.1
        ymin = np.min(pred_pose3d[:, :, 1]) - 0.1
        ymax = np.max(pred_pose3d[:, :, 1]) + 0.1
        zmin = np.min(pred_pose3d[:, :, 2]) - 0.1
        zmax = np.max(pred_pose3d[:, :, 2]) + 0.1

    def draw_skeleton(ax: plt.Axes, pred_pose3d: np.ndarray, gt_pose3d: np.ndarray | None = None) -> None:
        """Draw skeleton."""
        ax.clear()
        ax.set_title(title)
        ax.view_init(elev=-90, azim=-90)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if gt_pose3d is not None:
            gt_x0, gt_y0, gt_z0 = gt_pose3d[:, 0], gt_pose3d[:, 1], gt_pose3d[:, 2]
            ax.plot(gt_x0, gt_y0, gt_z0, "k.", label="GT", markersize=10, alpha=0.5)
            x_bone_gt, y_bone_gt, z_bone_gt = set_lines(gt_x0, gt_y0, gt_z0, h36m_bones)
            for x, y, z in zip(x_bone_gt, y_bone_gt, z_bone_gt, strict=False):
                line = art3d.Line3D(x, y, z, color="black", linewidth=4)
                ax.add_line(line)

        pred_x0, pred_y0, pred_z0 = pred_pose3d[:, 0], pred_pose3d[:, 1], pred_pose3d[:, 2]
        ax.plot(pred_x0, pred_y0, pred_z0, "r.", label="Pred", markersize=10, alpha=0.8)
        x_bone_pred, y_bone_pred, z_bone_pred = set_lines(pred_x0, pred_y0, pred_z0, h36m_bones)
        for x, y, z in zip(x_bone_pred, y_bone_pred, z_bone_pred, strict=False):
            line = art3d.Line3D(x, y, z, color="red", linewidth=4, alpha=0.8)
            ax.add_line(line)

    def update_frame(fc: int) -> None:
        """Update frame."""
        if gt_pose3d is not None:
            draw_skeleton(ax, pred_pose3d[fc], gt_pose3d[fc])
        else:
            draw_skeleton(ax, pred_pose3d[fc])

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=pred_pose3d.shape[0],
        interval=30,
        repeat=False,
    )
    if save_path is not None:
        anim.save(save_path, writer="ffmpeg", fps=30)
        plt.close(fig)
        del anim
    else:
        return anim
