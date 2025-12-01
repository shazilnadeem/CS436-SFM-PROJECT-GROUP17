import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags
import pillow_heif


def load_image(path: str):
    """
    Load an image from disk as a BGR OpenCV image.
    Handles HEIC/HEIF via pillow_heif, all others via cv2.imread.
    """
    ext = Path(path).suffix.lower()

    if ext in [".heic", ".heif"]:
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw"
        )

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def list_images(images_dir: str | Path, exts=None):
    """
    List image files in a directory, sorted by name.

    Args:
        images_dir: folder path
        exts: iterable of extensions to allow

    Returns:
        List[Path]
    """
    if exts is None:
        exts = [".jpg", ".jpeg", ".png", ".heic", ".heif"]

    images_dir = Path(images_dir)
    files = []
    for f in images_dir.iterdir():
        if f.suffix.lower() in exts:
            files.append(f)
    files.sort()
    return files


def get_intrinsics_from_exif(path, default_shape=None, fallback_factor=1.0):
    """
    Extract camera intrinsics (K matrix) from EXIF metadata if available.
    - For HEIC/HEIF: uses pillow_heif to load and attach EXIF
    - Tries to read 'FocalLengthIn35mmFilm' or 'FocalLength'
    - Converts focal length to pixel units
    - Falls back to focal_px = image_width * fallback_factor

    Returns:
        K : np.ndarray (3x3)
    """
    ext = path.lower().split('.')[-1]

    if ext in ['heic', 'heif']:
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, 'raw')
        if 'exif' in heif_file.info:
            img.info['exif'] = heif_file.info['exif']
    else:
        img = Image.open(path)

    w, h = img.size
    focal_px = None

    exif_data = None
    if hasattr(img, "_getexif"):
        raw = img._getexif()
        if raw:
            exif_data = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}

    if exif_data:
        if 'FocalLengthIn35mmFilm' in exif_data:
            f_35 = exif_data['FocalLengthIn35mmFilm']
            focal_px = (f_35 / 36.0) * w 
        elif 'FocalLength' in exif_data:
            f_mm = exif_data['FocalLength']
            if isinstance(f_mm, tuple):
                f_mm = float(f_mm[0]) / float(f_mm[1])
            print(f"[EXIF] Found FocalLength (mm) = {f_mm}; skipping direct mm->px conversion.")

    if focal_px is None:
        focal_px = w * fallback_factor

    K = np.array([
        [focal_px, 0,w / 2.0],
        [0,focal_px, h / 2.0],
        [0,0,1.0]
    ], dtype=np.float64)

    return K
