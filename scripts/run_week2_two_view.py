from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import load_image, list_images, get_intrinsics_from_exif
from src.vis_open3d import save_point_cloud, view_point_cloud

from src.io_utils import load_image, list_images, get_intrinsics_from_exif
from src.features import (
    create_detector,
    detect_features,
    match_features,
    matches_to_points,
)
from src.two_view import (
    estimate_essential,
    recover_pose_from_E,
    triangulate_points,
    filter_triangulated_points,
)
from src.vis_open3d import save_point_cloud


def main():
    images_dir = Path("data/images")
    image_paths = list_images(images_dir)

    if len(image_paths) < 6:
        raise RuntimeError("Need at least 6 images in data/images/ to use 5 & 6.")

    # Use 5th and 6th images (0-based indices 4 and 5)
    idx0, idx1 = 4, 5
    img1_path = image_paths[idx0]
    img2_path = image_paths[idx1]

    print(f"[week2] Using images {idx0}={img1_path.name}, {idx1}={img2_path.name}")

    img1 = load_image(str(img1_path))
    img2 = load_image(str(img2_path))

    detector = create_detector("SIFT")
    kp1, desc1 = detect_features(detector, img1)
    kp2, desc2 = detect_features(detector, img2)

    matches = match_features(desc1, desc2)
    print(f"[week2] {len(matches)} raw good matches.")

    pts1, pts2 = matches_to_points(kp1, kp2, matches)

    K = get_intrinsics_from_exif(str(img1_path), fallback_factor=1.0)

    # Essential with RANSAC
    E, mask_E = estimate_essential(K, pts1, pts2)
    inliers_E = mask_E.ravel().astype(bool)
    pts1_in = pts1[inliers_E]
    pts2_in = pts2[inliers_E]
    print(f"[week2] {pts1_in.shape[0]} inliers after Essential RANSAC.")

    if pts1_in.shape[0] < 8:
        raise RuntimeError("Not enough inliers after Essential RANSAC.")

    # Cheirality from recoverPose
    R, t, mask_pose = recover_pose_from_E(E, K, pts1_in, pts2_in)
    cheirality_mask = mask_pose.ravel().astype(bool)
    pts1_final = pts1_in[cheirality_mask]
    pts2_final = pts2_in[cheirality_mask]
    print(f"[week2] {pts1_final.shape[0]} points passed cheirality check.")

    if pts1_final.shape[0] < 8:
        raise RuntimeError("Not enough points after cheirality check.")

    # Triangulate + reprojection/depth filter
    pts3d = triangulate_points(K, R, t, pts1_final, pts2_final)
    pts3d_filt, mask_good = filter_triangulated_points(
        pts3d,
        pts1_final,
        pts2_final,
        K,
        R,
        t,
        max_reproj_error=5.0,
    )
    print(f"[week2] {pts3d.shape[0]} triangulated, {pts3d_filt.shape[0]} kept after reprojection+depth filter.")

    # map filtered points back to original matches to get colors
    inlier_ids = np.where(inliers_E)[0] 
    pose_ids = inlier_ids[cheirality_mask] 
    good_ids = pose_ids[mask_good] 

    h, w = img1.shape[:2]
    colors = []
    for mi in good_ids:
        kp = kp1[matches[mi].queryIdx]
        x, y = kp.pt
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            colors.append(img1[yi, xi].tolist())
        else:
            colors.append([255, 255, 255])
    colors = np.array(colors, dtype=np.float64)


    out_dir = Path("data/results/week2")
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pts3d_filt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=pts[:, 2],
        cmap="viridis",
        s=3,
        alpha=0.7,
    )
    ax.set_title("3D Point Cloud (Two-View Reconstruction)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Depth)")
    fig.colorbar(sc, ax=ax, label="Depth (Z)")

    fig.tight_layout()
    fig_3d_path = out_dir / "two_view_pointcloud_3d.png"
    fig.savefig(fig_3d_path, dpi=200)
    plt.close(fig)
    print(f"[week2] Saved 3D Matplotlib view to {fig_3d_path}")

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    z_vals = pts[:, 2]

    x_lo, x_hi = np.percentile(x_vals, [2, 98])
    y_lo, y_hi = np.percentile(y_vals, [2, 98])
    valid_xy = (
        (x_vals >= x_lo) & (x_vals <= x_hi) &
        (y_vals >= y_lo) & (y_vals <= y_hi)
    )

    x_plot = x_vals[valid_xy]
    y_plot = y_vals[valid_xy]
    z_plot = z_vals[valid_xy]

    fig2 = plt.figure(figsize=(9, 9))
    plt.scatter(
        x_plot, y_plot,
        s=6,
        c=z_plot,
        cmap="inferno",
        alpha=0.85,
    )
    plt.title("Front View (x–y Projection)", fontsize=16)
    plt.xlabel("X Axis", fontsize=13)
    plt.ylabel("Y Axis", fontsize=13)
    plt.xlim(x_plot.min(), x_plot.max())
    plt.ylim(y_plot.min(), y_plot.max())
    plt.gca().set_aspect("equal", "box")
    plt.grid(alpha=0.25)

    fig_xy_path = out_dir / "two_view_front_xy.png"
    plt.tight_layout()
    plt.savefig(fig_xy_path, dpi=200)
    plt.close(fig2)
    print(f"[week2] Saved front-view projection to {fig_xy_path}")

    out_ply = out_dir / "two_view_5_6.ply"
    save_point_cloud(pts3d_filt, colors, str(out_ply))
    print(f"[week2] Saved point cloud to {out_ply}")

    try:
        view_point_cloud(str(out_ply))
    except Exception as e:
        print(f"[week2] Open3D viewer failed: {e}")



if __name__ == "__main__":
    main()
