from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .io_utils import load_image
from .features import (
    create_detector,
    detect_features,
    match_features,
    matches_to_points,
)
from .two_view import (
    estimate_essential,
    recover_pose_from_E,
    triangulate_points,
)


@dataclass
class ImageFeatures:
    """
    Features for a single image: keypoints, descriptors, and original BGR image.
    """
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    img: np.ndarray


@dataclass
class SfMState:
    """
    Global state for incremental SfM reconstruction.
    World frame is defined by the bootstrap first camera.
    """
    K: np.ndarray    
    images: List[Path] = field(default_factory=list) 

    # camera poses in WORLD frame: X_cam = R * X_world + t
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    # 3D structure in WORLD frame
    points3d: List[List[float]] = field(default_factory=list)
    colors: List[List[float]] = field(default_factory=list)

    # tracks: (img_idx, kp_idx) -> point index
    tracks: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # precomputed features for each image
    features: Dict[int, ImageFeatures] = field(default_factory=dict)

    def add_pose(self, idx: int, R: np.ndarray, t: np.ndarray) -> None:
        """
        Store the pose of image idx in WORLD coordinates.
        """
        self.poses[idx] = (R.copy(), t.copy())

    def add_points(
        self,
        pts3d_world: np.ndarray,
        img_idx0: int,
        kp_indices0: List[int],
        img_idx1: int,
        kp_indices1: List[int],
        color_image: np.ndarray,
    ) -> None:
        """
        Insert new 3D points in WORLD frame and record 2D–3D associations
        for two images (img_idx0 and img_idx1).

        Colors are sampled from `color_image` at keypoints in img_idx0
        (kp_indices0)
        """
        pts3d_world = np.asarray(pts3d_world, dtype=np.float64)
        n_pts = pts3d_world.shape[0]
        if n_pts == 0:
            return

        assert n_pts == len(kp_indices0) == len(kp_indices1), \
            "Points and keypoint index lists must have same length."

        start_idx = len(self.points3d)
        self.points3d.extend(pts3d_world.tolist())

        # sample colors from img_idx0's image
        h, w = color_image.shape[:2]
        for kp_id in kp_indices0:
            kp = self.features[img_idx0].keypoints[kp_id]
            x, y = kp.pt
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                self.colors.append(color_image[yi, xi].tolist())
            else:
                # fallback color if out of bounds
                self.colors.append([255, 255, 255])

        # record 2D–3D tracks for both images
        for i in range(n_pts):
            point_index = start_idx + i
            self.tracks[(img_idx0, kp_indices0[i])] = point_index
            self.tracks[(img_idx1, kp_indices1[i])] = point_index

    def get_points_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all 3D points and colors as arrays.
        """
        if len(self.points3d) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
            )
        pts = np.array(self.points3d, dtype=np.float64)
        cols = np.array(self.colors, dtype=np.float64)
        return pts, cols


def compute_image_features(
    image_paths: List[Path],
    detector_method: str = "SIFT",
) -> Dict[int, ImageFeatures]:
    """
    Compute and store features for all images.
    """
    detector = create_detector(detector_method)
    features: Dict[int, ImageFeatures] = {}

    for idx, path in enumerate(image_paths):
        img = load_image(str(path))
        kp, desc = detect_features(detector, img)
        features[idx] = ImageFeatures(
            keypoints=kp,
            descriptors=desc,
            img=img,
        )
        print(f"[features] Image {idx}: {path.name}, keypoints={len(kp)}")

    return features


def bootstrap_two_view(
    state: SfMState,
    idx0: int,
    idx1: int,
    max_depth: float = 1000.0,
) -> None:
    """
    Initialize SfM from images idx0 and idx1:

      - SIFT + FLANN + ratio test matching
      - Essential matrix with RANSAC
      - recoverPose with cheirality mask
      - Triangulate cheirality inliers
      - Filter only by:
            * finite 3D
            * 0 < z < max_depth

    World frame = camera of idx0.
    """
    f0 = state.features[idx0]
    f1 = state.features[idx1]

    matches = match_features(f0.descriptors, f1.descriptors)
    print(f"[bootstrap] {len(matches)} raw matches between {idx0} and {idx1}")
    if len(matches) < 8:
        raise RuntimeError("Not enough matches for bootstrap two-view.")

    pts0, pts1 = matches_to_points(f0.keypoints, f1.keypoints, matches)

    E, mask_E = estimate_essential(state.K, pts0, pts1)
    inliers_E = mask_E.ravel().astype(bool)
    pts0_in = pts0[inliers_E]
    pts1_in = pts1[inliers_E]
    print(f"[bootstrap] {pts0_in.shape[0]} inliers after Essential RANSAC.")

    if pts0_in.shape[0] < 8:
        raise RuntimeError("Not enough inliers after Essential RANSAC.")

    R, t, mask_pose = recover_pose_from_E(E, state.K, pts0_in, pts1_in)
    cheirality_mask = mask_pose.ravel().astype(bool)
    pts0_ch = pts0_in[cheirality_mask]
    pts1_ch = pts1_in[cheirality_mask]
    print(f"[bootstrap] {pts0_ch.shape[0]} points passed cheirality check.")

    if pts0_ch.shape[0] < 8:
        raise RuntimeError("Not enough points after cheirality check in bootstrap.")


    pts3d = triangulate_points(state.K, R, t, pts0_ch, pts1_ch)


    valid_finite = np.isfinite(pts3d).all(axis=1)
    z = pts3d[:, 2]
    valid_depth = (z > 0.0) & (z < max_depth)
    valid_mask = valid_finite & valid_depth

    pts3d_filt = pts3d[valid_mask]
    print(f"[bootstrap] {pts3d.shape[0]} triangulated, {pts3d_filt.shape[0]} kept after depth/finite filter.")

    if pts3d_filt.shape[0] == 0:
        raise RuntimeError("No valid 3D points after filtering in bootstrap.")


    inlier_ids = np.where(inliers_E)[0] 
    pose_ids = inlier_ids[cheirality_mask] 
    good_ids = pose_ids[valid_mask] 

    kp_ids0 = [matches[i].queryIdx for i in good_ids]
    kp_ids1 = [matches[i].trainIdx for i in good_ids]


    state.add_pose(idx0, np.eye(3), np.zeros((3, 1)))
    state.add_pose(idx1, R, t)

    state.add_points(
        pts3d_world=pts3d_filt,
        img_idx0=idx0,
        kp_indices0=kp_ids0,
        img_idx1=idx1,
        kp_indices1=kp_ids1,
        color_image=f0.img,
    )
    print(f"[bootstrap] Bootstrap complete. Initial 3D points: {len(state.points3d)}")


def register_new_view(
    state: SfMState,
    new_idx: int,
    ref_idx: int,
    min_inliers: int = 8,
    max_depth: float = 1000.0,
) -> None:
    """
    Register a new view incrementally:

      1. Match features between ref image and new image
      2. Build 2D–3D correspondences from ref image tracks
      3. Run solvePnPRansac to get new camera pose in WORLD frame
      4. Triangulate NEW 3D points only from matches that do NOT
         already have 3D (matches_for_triangulation)
      5. Filter by finite, 0 < z < max_depth
      6. Add new 3D points, sampling COLORS from the NEW image

    Args:
        state: global SfM state
        new_idx: index of the new image to register
        ref_idx: index of the reference image (already in the map)
    """
    f_ref = state.features[ref_idx]
    f_new = state.features[new_idx]


    matches = match_features(f_ref.descriptors, f_new.descriptors)
    print(f"[register] {len(matches)} matches between ref={ref_idx} and new={new_idx}")

    if len(matches) < min_inliers:
        raise RuntimeError("Not enough descriptor matches between ref and new image.")

    object_points = []
    image_points = []
    matches_for_triangulation = []

    for m in matches:
        key_ref = (ref_idx, m.queryIdx)
        if key_ref in state.tracks:
            p_idx = state.tracks[key_ref]
            object_points.append(state.points3d[p_idx])
            image_points.append(f_new.keypoints[m.trainIdx].pt)
        else:
            matches_for_triangulation.append(m)

    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)

    print(f"[register] {object_points.shape[0]} 2D–3D correspondences for PnP.")
    if object_points.shape[0] < min_inliers:
        raise RuntimeError("Not enough 2D–3D correspondences for PnP.")

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        state.K,
        None,
    )
    if not success:
        raise RuntimeError("PnP failed for new view.")

    R_new, _ = cv2.Rodrigues(rvec)
    t_new = tvec.reshape(3, 1)
    print(f"[register] PnP inliers: {inliers.shape[0]}")

    state.add_pose(new_idx, R_new, t_new)

    if len(matches_for_triangulation) == 0:
        print("[register] No matches left for triangulation.")
        return

    pts_ref_tri = []
    pts_new_tri = []
    for m in matches_for_triangulation:
        pts_ref_tri.append(f_ref.keypoints[m.queryIdx].pt)
        pts_new_tri.append(f_new.keypoints[m.trainIdx].pt)

    pts_ref_tri = np.float32(pts_ref_tri)
    pts_new_tri = np.float32(pts_new_tri)

    if ref_idx not in state.poses:
        raise RuntimeError("Reference view has no stored pose.")
    R_ref, t_ref = state.poses[ref_idx]

    P_ref = state.K @ np.hstack((R_ref, t_ref))
    P_new = state.K @ np.hstack((R_new, t_new))

    pts4d = cv2.triangulatePoints(P_ref, P_new, pts_ref_tri.T, pts_new_tri.T)
    pts3d_world = (pts4d[:3] / pts4d[3]).T

    finite_mask = np.isfinite(pts3d_world).all(axis=1)
    z = pts3d_world[:, 2]
    depth_mask = z > 0.0
    norm_vals = np.linalg.norm(pts3d_world, axis=1)
    norm_mask = norm_vals < max_depth
    valid_mask = finite_mask & depth_mask & norm_mask

    if not np.any(valid_mask):
        print("[register] All triangulated points rejected by depth/finite filter.")
        return

    pts3d_valid = pts3d_world[valid_mask]

    tri_matches = np.array(matches_for_triangulation)
    valid_indices = np.where(valid_mask)[0]

    kp_ref_new = []
    kp_new_new = []

    for idx in valid_indices:
        m = tri_matches[idx]
        kp_ref_new.append(m.queryIdx)
        kp_new_new.append(m.trainIdx)

    state.add_points(
        pts3d_world=pts3d_valid,
        img_idx0=new_idx,
        kp_indices0=kp_new_new,
        img_idx1=ref_idx,
        kp_indices1=kp_ref_new,
        color_image=f_new.img,
    )

    print(f"[register] Added {pts3d_valid.shape[0]} new 3D points from view {new_idx}.")
