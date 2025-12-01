import numpy as np
import open3d as o3d


def _to_o3d_colors(colors: np.ndarray) -> np.ndarray:
    """
    Convert BGR [0..255] or RGB [0..255]/[0..1] into RGB [0..1]
    exactly like in the notebook.

    - If values > 1.0 → assume 0..255 and divide by 255.
    - If input is BGR, swap to RGB.
    """
    cols = np.asarray(colors, dtype=np.float64)

    if cols.ndim == 1:
        cols = cols[None, :]

    # Normalize if needed
    if cols.max() > 1.0:
        cols = cols / 255.0

    # If we have 3 channels, assume BGR (OpenCV) → RGB
    if cols.shape[1] == 3:
        cols = cols[:, [2, 1, 0]]  # swap B,G,R → R,G,B

    return cols


def save_point_cloud(points3d, colors, filename: str) -> None:
    """
    Save a colored point cloud to PLY.
    Points are N×3, colors are N×3 sampled from OpenCV images (BGR).

    The color conversion matches the notebook:
    BGR [0..255] → RGB [0..1].
    """
    pts = np.asarray(points3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points3d must be of shape (N, 3)")

    cols = _to_o3d_colors(colors)

    if cols.shape[0] != pts.shape[0]:
        raise ValueError("points3d and colors must have the same length")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"[io] Saved point cloud to {filename}")


def view_point_cloud(filename: str) -> None:
    """
    Open an interactive Open3D viewer for a saved PLY file
    with a BLACK background (matching typical SfM viewers).
    """
    pcd = o3d.io.read_point_cloud(filename)
    if pcd.is_empty():
        print("[io] Warning: point cloud is empty.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="SfM Point Cloud",
        width=1024,
        height=768,
        visible=True,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 3.0 

    vis.run()
    vis.destroy_window()

