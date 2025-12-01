import cv2
import numpy as np


def build_K_from_shape(image_shape):
    """
    Approximate intrinsics from image shape:
    fx = fy = width, cx = w/2, cy = h/2.
    (Used if EXIF-based K is not available.)
    """
    h, w = image_shape[:2]
    f = w
    K = np.array(
        [
            [f, 0,w / 2.0],
            [0, f,h / 2.0],
            [0, 0,1.0],
        ],
        dtype=np.float64,
    )
    return K


def estimate_essential(K, pts1, pts2, threshold=1.0, prob=0.999):
    """
    Estimate Essential matrix using RANSAC
    """
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=prob,
        threshold=threshold,
    )
    return E, mask


def recover_pose_from_E(E, K, pts1, pts2):
    """
    Recover a single (R, t) from E and matched points,
    using cv2.recoverPose
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask


def triangulate_points(K, R, t, pts1, pts2):
    """
    Triangulate 3D points from two calibrated views, assuming
    world frame = first camera with [I | 0] and second camera with [R | t].
    """
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])

    pts1_T = pts1.T
    pts2_T = pts2.T

    pts4d = cv2.triangulatePoints(P0, P1, pts1_T, pts2_T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d


def reprojection_errors(pts3d, K, R, t, pts1, pts2):
    """
    Helper for Week 2-type filtering. Not used in the final Week 3 pipeline,
    but kept for completeness / reuse.
    """
    pts_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))
    proj0 = (P0 @ pts_h.T).T
    proj1 = (P1 @ pts_h.T).T

    proj0 = proj0[:, :2] / np.clip(proj0[:, 2:3], 1e-6, None)
    proj1 = proj1[:, :2] / np.clip(proj1[:, 2:3], 1e-6, None)

    err0 = np.linalg.norm(proj0 - pts1, axis=1)
    err1 = np.linalg.norm(proj1 - pts2, axis=1)
    return (err0 + err1) / 2.0


def filter_triangulated_points(
    pts3d,
    pts1,
    pts2,
    K,
    R,
    t,
    max_reproj_error=5.0,
    min_depth=1e-3,
):
    """
    filtering:
      - remove NaN/inf 3D points
      - reprojection error filter
      - enforce positive depth in both cameras
    """
    # 1) Drop NaN / inf points
    valid_mask = np.isfinite(pts3d).all(axis=1)
    pts3d = pts3d[valid_mask]
    pts1 = pts1[valid_mask]
    pts2 = pts2[valid_mask]

    err = reprojection_errors(pts3d, K, R, t, pts1, pts2)

    z0 = pts3d[:, 2]
    pts3d_cam1 = (R @ pts3d.T + t).T
    z1 = pts3d_cam1[:, 2]

    mask = (err < max_reproj_error) & (z0 > min_depth) & (z1 > min_depth)
    return pts3d[mask], mask
