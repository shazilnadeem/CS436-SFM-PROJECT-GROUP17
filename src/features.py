import cv2
import numpy as np


def create_detector(method: str = "SIFT"):
    m = method.upper()
    if m == "SIFT":
        return cv2.SIFT_create(nfeatures=5000)
    elif m == "ORB":
        return cv2.ORB_create()
    else:
        raise ValueError(f"Unknown feature method: {method}")



def detect_features(detector, img_bgr):
    """
    Detect keypoints and descriptors on a BGR image.

    Returns:
        keypoints, descriptors
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio: float = 0.75):
    """
    Match descriptors using FLANN + Lowe's ratio test.

    Returns:
        List of cv2.DMatch
    """
    if desc1 is None or desc2 is None:
        return []

    if desc1.dtype != np.float32:
        desc1 = desc1.astype(np.float32)
    if desc2.dtype != np.float32:
        desc2 = desc2.astype(np.float32)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def matches_to_points(kp1, kp2, matches):
    """
    Convert matches into Nx2 coordinate arrays (pts1, pts2).
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def draw_matches(img1, kp1, img2, kp2, matches, max_draw=None):
    """
    Draw matches using cv2.drawMatches exactly like your notebook version.
    """
    if max_draw is not None:
        matches = matches[:max_draw]

    out = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return out
