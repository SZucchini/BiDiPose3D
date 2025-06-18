"""Sampler for stereo camera positions and target positions."""

import numpy as np


class StereoCameraSampler:
    """Sample stereo camera positions and target positions."""

    def __init__(
        self,
        dmin_factor: float = 1.5,
        dmax_factor: float = 3.0,
        baseline_angle_range: tuple[float, float] = (60.0, 120.0),
        rng: np.random.Generator | None = None,
        max_iter: int = 10000,
    ) -> None:
        """Initialize the sampler.

        Args:
            distance_factor (float): Factor to determine camera distance from the scene.
            baseline_angle_range (tuple[float, float]): Range of angles for the baseline between cameras in degrees.
            height_tol_ratio (float): Ratio for height tolerance between cameras.
            min_polar_angle_deg (float): Minimum polar angle in degrees for camera placement.
            rng (np.random.Generator | None): Random number generator. If None, uses the default RNG.
            max_iter (int): Maximum number of iterations to sample camera and target positions.

        """
        self.dmin_factor = dmin_factor
        self.dmax_factor = dmax_factor
        self.baseline_angle_range = baseline_angle_range
        self.img_size_options = [
            (3840, 2160),
            (1920, 1080),
            (1280, 720),
            (1000, 1000),
        ]
        self.fov_range_deg = (40.0, 120.0)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.max_iter = max_iter

    def _sample_intrinsics(self, image_size: tuple[int, int], h_fov_rad: float) -> np.ndarray:
        """Sample camera intrinsics."""
        w, h = image_size
        base_f = (w / 2) / np.tan(h_fov_rad / 2)
        fx = base_f * self.rng.uniform(0.95, 1.05)
        fy = base_f * self.rng.uniform(0.95, 1.05)
        cx = w / 2 + self.rng.normal(0, 0.01 * w)
        cy = h / 2 + self.rng.normal(0, 0.01 * h)
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        return intrinsics

    def _compute_rotation(self, cam: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        up_hint = np.array([0.0, 0.0, 1.0], float)
        z_cam = tgt - cam
        z_cam /= np.linalg.norm(z_cam)
        if abs(np.dot(z_cam, up_hint) / np.linalg.norm(up_hint)) > 0.99:
            up_hint = np.array([0.0, 1.0, 0.0], float)
        x_cam = np.cross(z_cam, up_hint)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        return np.stack((x_cam, y_cam, z_cam), 0)

    def _check_is_inside_image(self, kpts_world: np.ndarray, cam_pos: np.ndarray, tgt_pos: np.ndarray) -> bool:
        """Check if the keypoints are inside the image."""
        w, h = self.img_size
        for cam_idx in range(2):
            rot = self._compute_rotation(cam_pos[cam_idx], tgt_pos[cam_idx])
            kpts_pixel = self._world_to_image(kpts_world, cam_pos[cam_idx], rot, self.intrinsics[cam_idx])
            kpts_pixel = kpts_pixel.reshape(-1, 2)
            x_min, x_max = kpts_pixel[:, 0].min(), kpts_pixel[:, 0].max()
            y_min, y_max = kpts_pixel[:, 1].min(), kpts_pixel[:, 1].max()
            if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
                return False
        return True

    def _world_to_camera(self, kpts_world: np.ndarray, cam: np.ndarray, rot: np.ndarray) -> np.ndarray:
        p = np.asarray(kpts_world, float).reshape(-1, 3)
        p_cam = (p - cam) @ rot.T
        return p_cam.reshape(kpts_world.shape)

    def _camera_to_image(self, kpts_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        p = np.asarray(kpts_cam, float).reshape(-1, 3)
        x = p[:, 0] / p[:, 2]
        y = p[:, 1] / p[:, 2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = fx * x + cx
        v = fy * y + cy
        return np.stack((u, v), -1).reshape(kpts_cam.shape[:-1] + (2,))

    def _world_to_image(
        self, kpts_world: np.ndarray, cam_pos: np.ndarray, rot: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        return self._camera_to_image(self._world_to_camera(kpts_world, cam_pos, rot), intrinsics)

    def _lift(self, ground_vec: np.ndarray, elev_rad: float) -> np.ndarray:
        d = np.linalg.norm(ground_vec)
        if d == 0:
            return ground_vec
        horiz = ground_vec * np.cos(elev_rad)
        vert = np.array([0.0, 0.0, d * np.sin(elev_rad)], float)
        return horiz + vert

    def sample_camera_and_target(self, kpts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample camera and target positions based on the keypoints in the world coordinate system.

        Args:
            kpts_world (np.ndarray): Keypoints in the world coordinate system. The shape is (T, J, 3).

        Returns:
            tuple[np.ndarray, np.ndarray]: Camera positions (shape (2, 3)) and target positions (shape (2, 3)).

        """
        max_retry = 1000
        for retry_count in range(max_retry):
            try:
                pts = kpts_world.reshape(-1, 3)
                centre = 0.5 * (pts.min(0) + pts.max(0))
                radius = np.linalg.norm(pts - centre, axis=1).max()

                self.img_size = self.rng.choice(self.img_size_options)
                aspect = self.img_size[0] / self.img_size[1]
                h_fov = np.deg2rad(self.rng.uniform(*self.fov_range_deg))
                v_fov = 2 * np.arctan(np.tan(h_fov / 2) / aspect)
                self.intrinsics = np.stack([self._sample_intrinsics(self.img_size, h_fov) for _ in range(2)])

                min_fov = min(h_fov, v_fov)
                d_min = max(radius * self.dmin_factor, radius / np.tan(min_fov / 2))
                d_max = radius * self.dmax_factor
                if d_min > d_max:
                    d_min, d_max = d_max, d_min
                baseline_min, baseline_max = map(np.deg2rad, self.baseline_angle_range)
                for _ in range(self.max_iter):
                    d1, d2 = self.rng.uniform(d_min, d_max, size=2)
                    phi1 = self.rng.uniform(0, 2 * np.pi)
                    g_vec1 = np.array([np.cos(phi1), np.sin(phi1), 0.0], float) * d1
                    theta1 = np.deg2rad(self.rng.uniform(-5.0, 45.0))
                    cam1 = centre + self._lift(g_vec1, theta1)

                    delta = self.rng.uniform(baseline_min, baseline_max)
                    if self.rng.random() < 0.5:
                        delta = -delta
                    phi2 = phi1 + delta
                    g_vec2 = np.array([np.cos(phi2), np.sin(phi2), 0.0], float) * d2
                    theta2 = np.deg2rad(self.rng.uniform(-5.0, 45.0))
                    cam2 = centre + self._lift(g_vec2, theta2)

                    dirs = self.rng.normal(0.0, 1.0, size=(2, 3))
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    dists = self.rng.uniform(0.0, radius, size=(2, 1))
                    tgts = centre + dirs * dists

                    if self._check_is_inside_image(kpts_world, np.stack((cam1, cam2)), tgts):
                        return np.stack((cam1, cam2)).astype(np.float32), tgts.astype(np.float32)
                    else:
                        continue

            except Exception as e:
                if retry_count == max_retry - 1:
                    raise RuntimeError(
                        f"Failed to sample camera and target positions after {max_retry} retry attempts. Last error: {str(e)}"
                    )

        raise RuntimeError(
            f"Failed to sample camera and target positions within the maximum number of iterations ({self.max_iter}) after {max_retry} retry attempts."
        )

    def _rot_to_quat(self, rot: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion.

        Args:
            rot (np.ndarray): Rotation matrix of shape (3, 3).

        Returns:
            np.ndarray: Quaternion of shape (4,).

        """
        q = np.empty(4, float)
        tr = np.trace(rot)
        if tr > 0.0:
            s = 0.5 / np.sqrt(tr + 1.0)
            q[0] = 0.25 / s
            q[1] = (rot[2, 1] - rot[1, 2]) * s
            q[2] = (rot[0, 2] - rot[2, 0]) * s
            q[3] = (rot[1, 0] - rot[0, 1]) * s
        else:
            i = np.argmax(np.diag(rot))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
            q[0] = (rot[k, j] - rot[j, k]) / s
            q[i + 1] = 0.25 * s
            q[j + 1] = (rot[j, i] + rot[i, j]) / s
            q[k + 1] = (rot[k, i] + rot[i, k]) / s
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        return q

    def _quat_to_rot(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Quaternion of shape (4,).

        Returns:
            np.ndarray: Rotation matrix of shape (3, 3).

        """
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            float,
        )

    def get_relative_pose(self, cam_pos: np.ndarray, tgt_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get the relative pose of the cameras and target positions.

        Args:
            cam_pos (np.ndarray): Camera positions of shape (2, 3).
            tgt_pos (np.ndarray): Target positions of shape (2, 3).

        Returns:
            tuple[np.ndarray, np.ndarray]: Quaternion representing the rotation from camera 1 to camera 2
                and translation vector from camera 1 to camera 2.

        """
        rot1 = self._compute_rotation(cam_pos[0], tgt_pos[0])
        rot2 = self._compute_rotation(cam_pos[1], tgt_pos[1])
        rot_rel = rot2 @ rot1.T
        quat_rel = self._rot_to_quat(rot_rel)

        t_rel = rot2 @ (cam_pos[0] - cam_pos[1])
        t_rel /= np.linalg.norm(t_rel)
        return quat_rel, t_rel

    def triangulate(
        self,
        uv1: np.ndarray,
        uv2: np.ndarray,
        quat_rel: np.ndarray,
        t_rel: np.ndarray,
        baseline_len: float | None = None,
    ) -> np.ndarray:
        """Triangulate 3D points from stereo pixel coordinates and relative pose.

        Args:
            uv1 (np.ndarray): Pixel coordinates in camera-1 image, shape (..., 2).
            uv2 (np.ndarray): Pixel coordinates in camera-2 image, shape (..., 2).
            quat_rel (np.ndarray): Quaternion representing the rotation from camera-1 to camera-2, shape (4,).
            t_rel (np.ndarray): Translation vector from camera-1 to camera-2, shape (3,).
            baseline_len (float | None): Length of the baseline between cameras. If None, defaults to 1.0.

        Returns:
            np.ndarray: Triangulated 3D points in camera-1 coordinates, shape (..., 3).

        """
        intr1 = self.intrinsics[0]
        intr2 = self.intrinsics[1]
        if baseline_len is None:
            baseline_len = 1.0

        def _bearing(uv, intrinsics):
            uv = np.asarray(uv, float)
            ones = np.ones((*uv.shape[:-1], 1), uv.dtype)
            xyz = (np.linalg.inv(intrinsics) @ np.concatenate([uv, ones], -1)[..., None]).squeeze(-1)
            return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)

        v1 = _bearing(uv1, intr1)
        v2_c2 = _bearing(uv2, intr2)

        rot = self._quat_to_rot(quat_rel)
        v2 = (rot.T @ v2_c2[..., None]).squeeze(-1)
        o2 = -rot.T @ t_rel * baseline_len

        a = (v1 * v1).sum(-1, keepdims=True)
        b = (v1 * v2).sum(-1, keepdims=True)
        c = (v2 * v2).sum(-1, keepdims=True)
        w0 = -o2
        d = (v1 * w0).sum(-1, keepdims=True)
        e = (v2 * w0).sum(-1, keepdims=True)
        denom = a * c - b * b + 1e-12

        s1 = (b * e - c * d) / denom
        s2 = (a * e - b * d) / denom

        p1 = v1 * s1
        p2 = o2 + v2 * s2
        pts = 0.5 * (p1 + p2)
        return pts

    def project_and_normalize(
        self,
        kpts_world: np.ndarray,
        cam_pos: np.ndarray,
        tgt_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project keypoints from world coordinates to image coordinates and normalize them.

        Args:
            kpts_world (np.ndarray): Keypoints in world coordinates, shape (T, J, 3).
            cam_pos (np.ndarray): Camera positions, shape (2, 3).
            tgt_pos (np.ndarray): Target positions, shape (2, 3).

        Returns:
            xn1 (np.ndarray): Normalized keypoints in camera-1 coordinates, shape (T, J, 3).
            xn2 (np.ndarray): Normalized keypoints in camera-2 coordinates, shape (T, J, 3).

        """
        rot1 = self._compute_rotation(cam_pos[0], tgt_pos[0])
        rot2 = self._compute_rotation(cam_pos[1], tgt_pos[1])

        uv1 = self._world_to_image(kpts_world, cam_pos[0], rot1, self.intrinsics[0])
        uv2 = self._world_to_image(kpts_world, cam_pos[1], rot2, self.intrinsics[1])

        def _normalize(uv, intrinsics):
            flat = uv.reshape(-1, 2)
            ones = np.ones((flat.shape[0], 1), flat.dtype)
            homog = np.hstack([flat, ones])
            xyz = (np.linalg.inv(intrinsics) @ homog.T).T
            xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
            return xyz.reshape(uv.shape[:-1] + (3,))

        xn1 = _normalize(uv1, self.intrinsics[0])
        xn2 = _normalize(uv2, self.intrinsics[1])
        return xn1, xn2
