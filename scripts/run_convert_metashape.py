from pathlib import Path
from src.convert_metashape_xml import export_cameras_json

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    xml_path = root / "data/metashape/exports/cameras_corridor.xml"
    out_json = root / "data/results/final/cameras_corridor.json"
    export_cameras_json(xml_path, out_json)
