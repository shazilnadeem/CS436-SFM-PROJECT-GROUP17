from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.io_utils import list_images, get_intrinsics_from_exif
from src.multiview_sfm import (
    SfMState,
    compute_image_features,
    bootstrap_two_view,
    register_new_view,
)
from src.vis_open3d import save_point_cloud, view_point_cloud


def main():
    images_dir = Path("data/images")
    image_paths = list_images(images_dir)

    if len(image_paths) < 2:
        raise RuntimeError("Need at least 2 images in data/images/ to bootstrap.")

    # Use 1st and 2nd images (0-based indices 0 and 1) as bootstrap pair,
    idx0, idx1 = 0, 1
    print(f"[week3] Bootstrap pair: {idx0}={image_paths[idx0].name}, {idx1}={image_paths[idx1].name}")

    # Build intrinsics from first bootstrap image via EXIF
    K = get_intrinsics_from_exif(str(image_paths[idx0]), fallback_factor=1.0)

    # Initialize SfM state
    state = SfMState(K=K, images=image_paths)

    # Precompute features for all images (SIFT nfeatures=5000 inside create_detector("SIFT"))
    state.features = compute_image_features(image_paths, detector_method="SIFT")

    # phase 1: bootstrap with two-view reconstruction
    bootstrap_two_view(state, idx0, idx1)

    # phase 2: incrementally register remaining images, previous -> current
    last_idx = idx1
    for new_idx in range(len(image_paths)):
        if new_idx in (idx0, idx1):
            continue
        print(f"[week3] Registering new view {new_idx}: {image_paths[new_idx].name}")
        try:
            register_new_view(
                state,
                new_idx,
                ref_idx=last_idx,
            )
            last_idx = new_idx
        except RuntimeError as e:
            print(f"[week3] Skipping view {new_idx} due to error: {e}")

    # phase 3: export final 3D map + visualisations
    pts3d, colors = state.get_points_array()
    print(f"[week3] Final map has {pts3d.shape[0]} points.")

    # Mild outlier removal
    MAX_OUTLIERS = 50

    if pts3d.shape[0] > MAX_OUTLIERS:
        norms = np.linalg.norm(pts3d, axis=1)   # distance from origin
        sorted_idx = np.argsort(norms)          # ascending
        keep_idx = sorted_idx[:-MAX_OUTLIERS]   # keep all except last N

        before = pts3d.shape[0]
        pts3d = pts3d[keep_idx]
        colors = colors[keep_idx]
        after = pts3d.shape[0]

        print(f"[week3] Mild outlier removal (top {MAX_OUTLIERS} farthest): {before} → {after}")


    out_dir = Path("data/results/week3")
    out_dir.mkdir(parents=True, exist_ok=True)

    if pts3d.shape[0] == 0:
        print("[week3] No points reconstructed, nothing to save/visualize.")
        return

    pts = pts3d

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=pts[:, 2],
        cmap="viridis",
        s=2,
        alpha=0.7,
    )
    ax.set_title("3D Point Cloud (Multi-View Reconstruction)", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Depth)")
    fig.colorbar(sc, ax=ax, label="Depth (Z)")
    fig.tight_layout()

    fig_3d_path = out_dir / "week3_pointcloud_3d.png"
    fig.savefig(fig_3d_path, dpi=200)
    plt.close(fig)
    print(f"[week3] Saved 3D Matplotlib view to {fig_3d_path}")

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    z_vals = pts[:, 2]

    # percentile clipping for outliers
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
        s=4,
        c=z_plot,
        cmap="inferno",
        alpha=0.85,
    )
    plt.title("Front View (x–y Projection, Multi-View)", fontsize=16)
    plt.xlabel("X Axis", fontsize=13)
    plt.ylabel("Y Axis", fontsize=13)
    plt.xlim(x_plot.min(), x_plot.max())
    plt.ylim(y_plot.min(), y_plot.max())
    plt.gca().set_aspect("equal", "box")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    fig_xy_path = out_dir / "week3_front_xy.png"
    plt.savefig(fig_xy_path, dpi=200)
    plt.close(fig2)
    print(f"[week3] Saved front-view projection to {fig_xy_path}")

    out_ply = out_dir / "week3_multiview.ply"
    save_point_cloud(pts3d, colors, str(out_ply))
    print(f"[week3] Saved multi-view point cloud to {out_ply}")

    try:
        view_point_cloud(str(out_ply))
    except Exception as e:
        print(f"[week3] Open3D viewer failed: {e}")


if __name__ == "__main__":
    main()
