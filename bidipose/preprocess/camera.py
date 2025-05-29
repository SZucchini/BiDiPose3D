"""Stereo-pair “look-at” pin-hole camera (OpenCV convention)."""

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix [v]ₓ."""
    return np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
        dtype=float,
    )


class LookAtCamera:
    """Two-camera “look-at” rig."""

    def __init__(
        self,
        camera_positions: np.ndarray,
        target_positions: np.ndarray,
        image_size: tuple[int, int] = (1920, 1080),
        intrinsics: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        up_hint: np.ndarray | None = None,
    ) -> None:
        """Initialize the LookAtCamera with camera and target positions.

        Args:
            camera_positions (np.ndarray): Camera positions in world coordinates. Shape (2, 3).
            target_positions (np.ndarray): Target positions in world coordinates. Shape (2, 3).
            image_size (tuple[int, int]): Image size (width, height). Defaults to (1920, 1080).
            intrinsics (np.ndarray | None): Camera intrinsics matrix. If None, will be sampled.
            rng (np.random.Generator | None): Random number generator. Defaults to None.
            up_hint (np.ndarray | None): Hint for the up direction of the camera. Defaults to None.

        """
        cam_positions = np.asarray(camera_positions, float).reshape(-1, 3)
        tgt_positions = np.asarray(target_positions, float).reshape(-1, 3)
        if cam_positions.shape[0] != 2 or tgt_positions.shape[0] != 2:
            raise ValueError("This class is strictly limited to two cameras.")
        self.cam_pos: np.ndarray = cam_positions
        self.tgt_pos: np.ndarray = tgt_positions
        self.image_size = image_size
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        if intrinsics is None:
            self.K: np.ndarray = np.stack([self._sample_intrinsics(image_size) for _ in range(2)])
        else:
            self.K = intrinsics

        self.R: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        for cam_idx in range(2):
            rot = self._compute_rotation(cam_idx, up_hint)
            self.R.append(rot)
            self.t.append(-rot @ self.cam_pos[cam_idx])

    def _sample_intrinsics(self, image_size: tuple[int, int]) -> np.ndarray:
        """Sample camera intrinsics based on the image size.

        Args:
            image_size (tuple[int, int]): Image size (width, height).

        Returns:
            intrinsics (np.ndarray): Intrinsics matrix of shape (3, 3).

        """
        w, h = image_size
        fx = self.rng.uniform(8e2, 1.6e3)
        fy = fx * self.rng.uniform(0.95, 1.05)
        cx = w / 2 + self.rng.normal(0, 0.01 * w)
        cy = h / 2 + self.rng.normal(0, 0.01 * h)
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        return intrinsics

    def world_to_camera(self, xyz_w: np.ndarray, cam_idx: int) -> np.ndarray:
        """Convert world coordinates to camera coordinates.

        Args:
            xyz_w (np.ndarray): World coordinates of shape (..., 3).
            cam_idx (int): Index of the camera (0 or 1).

        Returns:
            np.ndarray: Camera coordinates of shape (..., 3).

        """
        p = np.asarray(xyz_w, float).reshape(-1, 3)
        p_cam = (p - self.cam_pos[cam_idx]) @ self.R[cam_idx].T
        return p_cam.reshape(xyz_w.shape)

    def camera_to_world(self, xyz_c: np.ndarray, cam_idx: int) -> np.ndarray:
        """Convert camera coordinates to world coordinates.

        Args:
            xyz_c (np.ndarray): Camera coordinates of shape (..., 3).
            cam_idx (int): Index of the camera (0 or 1).

        Returns:
            np.ndarray: World coordinates of shape (..., 3).

        """
        p = np.asarray(xyz_c, float).reshape(-1, 3)
        p_w = p @ self.R[cam_idx] + self.cam_pos[cam_idx]
        return p_w.reshape(xyz_c.shape)

    def camera_to_image(self, xyz_c: np.ndarray, cam_idx: int) -> np.ndarray:
        """Convert camera coordinates to image pixel coordinates.

        Args:
            xyz_c (np.ndarray): Camera coordinates of shape (..., 3).
            cam_idx (int): Index of the camera (0 or 1).

        Returns:
            np.ndarray: Image pixel coordinates of shape (..., 2).

        """
        p = np.asarray(xyz_c, float).reshape(-1, 3)
        x = p[:, 0] / p[:, 2]
        y = p[:, 1] / p[:, 2]
        fx, fy = self.K[cam_idx][0, 0], self.K[cam_idx][1, 1]
        cx, cy = self.K[cam_idx][0, 2], self.K[cam_idx][1, 2]
        u = fx * x + cx
        v = fy * y + cy
        return np.stack((u, v), -1).reshape(xyz_c.shape[:-1] + (2,))

    def world_to_image(self, xyz_w: np.ndarray, cam_idx: int) -> np.ndarray:
        """Convert world coordinates to image pixel coordinates.

        Args:
            xyz_w (np.ndarray): World coordinates of shape (..., 3).
            cam_idx (int): Index of the camera (0 or 1).

        Returns:
            np.ndarray: Image pixel coordinates of shape (..., 2).

        """
        return self.camera_to_image(self.world_to_camera(xyz_w, cam_idx), cam_idx)

    def pixel_to_normalized(self, uv: np.ndarray, cam_idx: int, homogeneous: bool = True) -> np.ndarray:
        """Convert pixel coordinates to normalized coordinates.

        Args:
            uv (np.ndarray): Pixel coordinates of shape (frames, joints, 2).
            cam_idx (int): Index of the camera (0 or 1).
            homogeneous (bool): If True, return homogeneous coordinates. Defaults to True.

        Returns:
            np.ndarray: Normalized coordinates of shape (frames, joints, 3) or (frames, joints, 2).

        """
        frames, joints, _ = uv.shape
        uv = np.asarray(uv, float).reshape(-1, 2)
        intrinsics_inv = np.linalg.inv(self.K[cam_idx])
        uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
        xyz_n = (intrinsics_inv @ uv_h.T).T
        if homogeneous:
            return xyz_n.reshape((frames, joints, 3))
        else:
            return xyz_n[:, :2].reshape((frames, joints, 2))

    def _compute_rotation(self, cam_idx: int, up_hint: np.ndarray | None) -> np.ndarray:
        """Compute the rotation matrix for the camera.

        Args:
            cam_idx (int): Index of the camera (0 or 1).
            up_hint (np.ndarray | None): Hint for the up direction of the camera. Defaults to None.

        Returns:
            np.ndarray: Rotation matrix of shape (3, 3).

        """
        if up_hint is None:
            up_hint = np.array([0.0, 0.0, 1.0], float)

        z_cam = self.tgt_pos[cam_idx] - self.cam_pos[cam_idx]
        z_cam /= np.linalg.norm(z_cam)

        if abs(np.dot(z_cam, up_hint) / np.linalg.norm(up_hint)) > 0.99:
            up_hint = np.array([0.0, 1.0, 0.0], float)

        x_cam = np.cross(z_cam, up_hint)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        return np.stack((x_cam, y_cam, z_cam), 0)
