from pathlib import Path

import cv2

from src.io_utils import load_image, list_images
from src.features import (
    create_detector,
    detect_features,
    match_features,
    draw_matches,
)


def main():
    images_dir = Path("data/images")
    image_paths = list_images(images_dir)

    if len(image_paths) < 6:
        raise RuntimeError("Need at least 6 images in data/images/ to use 5 & 6.")

    # Use 5th and 6th images (0-based indices 4 and 5)
    idx0, idx1 = 4, 5
    img1_path = image_paths[idx0]
    img2_path = image_paths[idx1]

    print(f"[week1] Using images {idx0}={img1_path.name}, {idx1}={img2_path.name}")

    img1 = load_image(str(img1_path))
    img2 = load_image(str(img2_path))

    detector = create_detector("SIFT")
    kp1, desc1 = detect_features(detector, img1)
    kp2, desc2 = detect_features(detector, img2)

    matches = match_features(desc1, desc2)
    print(f"[week1] {len(matches)} good matches between {img1_path.name} and {img2_path.name}")

    vis = draw_matches(img1, kp1, img2, kp2, matches, max_draw=None)

    out_dir = Path("data/results/week1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    out_path = out_dir / "week1_matches_5_6.jpg"

    cv2.imwrite(str(out_path), vis)
    print(f"[week1] Saved visualization to {out_path}")



if __name__ == "__main__":
    main()
