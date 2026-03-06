from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def bilateral_filter_cv(
    image_bgr: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 45.0,
    sigma_space: float = 45.0,
) -> np.ndarray:
    """Simple wrapper around OpenCV's built-in bilateral filter.

    This version is much faster than :func:`bilateral_filter_manual` and
    should be preferred in production. The manual implementation is left for
    reference or for experimenting with custom kernels.
    """
    return cv2.bilateralFilter(image_bgr, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def clahe_cv(
    l_channel: np.ndarray,
    tile_grid_size: tuple[int, int] = (8, 8),
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Apply CLAHE using OpenCV's factory method.

    The manual version :func:`clahe_manual` duplicates the internal logic and
    can be used for learning or small modifications, but the built-in is
    optimized and should be the default.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(l_channel)


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a BGR image for further analysis.

    The original implementation used hand-crafted versions of bilateral
    filtering and CLAHE.  Here we call the direct OpenCV methods, falling
    back to the manual ones only if needed.
    """

    bilateral = bilateral_filter_cv(
        image_bgr=image_bgr, diameter=5, sigma_color=45.0, sigma_space=45.0
    )

    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    enhanced_l = clahe_cv(l_channel, tile_grid_size=(8, 8), clip_limit=2.0)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_file(input_path: Path, output_path: Path) -> bool:
    image = cv2.imread(str(input_path))
    if image is None:
        return False

    processed = preprocess_image(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), processed))
