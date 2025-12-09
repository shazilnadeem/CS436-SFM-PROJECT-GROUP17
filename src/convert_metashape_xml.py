from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import json


def _q(root, tag):
    """Handle optional XML namespaces."""
    if "}" in root.tag:
        ns = root.tag.split("}")[0].strip("{")
        return f"{{{ns}}}{tag}"
    return tag


def load_metashape_cameras(xml_path: Path):
    """
    Parse Metashape cameras.xml and return a list of dicts.

    Assumes each <camera> has a <transform> with 16 numbers (4x4 matrix, row-major).
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    q = lambda t: _q(root, t)

    cameras = []
    for cam_el in root.findall(".//" + q("camera")):
        label = cam_el.get("label") or cam_el.get("id")
        transform_el = cam_el.find(q("transform"))
        if transform_el is None or transform_el.text is None:
            continue

        vals = [float(v) for v in transform_el.text.split()]
        if len(vals) != 16:
            continue

        T = np.array(vals, dtype=float).reshape(4, 4)

        position = T[:3, 3].tolist()
        rotation = T[:3, :3].tolist()

        cameras.append(
            {
                "id": len(cameras),
                "image_name": label,
                "matrix4x4": T.tolist(),
                "position": position,
                "rotation": rotation,
            }
        )

    return cameras


def export_cameras_json(xml_path: Path, out_json: Path):
    cams = load_metashape_cameras(xml_path)
    payload = {"cameras": cams}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[convert] Parsed {len(cams)} cameras")
    print(f"[convert] Wrote {out_json}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    xml_path = project_root / "data/metashape/exports/cameras_corridor.xml"
    out_json = project_root / "results/final/cameras_corridor.json"
    export_cameras_json(xml_path, out_json)
