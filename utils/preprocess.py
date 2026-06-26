# ============================================================
# FILE: utils/preprocess.py
# PURPOSE: Enhance plate images before OCR
# EXPLANATION:
#   OCR works much better on clean, high-contrast images.
#   This module applies computer vision tricks to:
#     - Remove noise
#     - Increase contrast
#     - Sharpen text edges
#   Better preprocessed image → more accurate OCR
# ============================================================

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """
    Master preprocessing pipeline for a cropped plate image.
    Applies all enhancement steps in sequence.

    Args:
        image: BGR or grayscale crop of the plate

    Returns:
        Enhanced grayscale image ready for OCR
    """
    if image is None or image.size == 0:
        logger.warning("Empty image passed to preprocessor.")
        return image

    try:
        # Step 1: Convert to grayscale
        # OCR doesn't need color — grayscale is simpler and faster
        gray = to_grayscale(image)

        # Step 2: Resize if too small
        # EasyOCR struggles with very small plates
        gray = resize_if_small(gray, min_width=100)

        # Step 3: Denoise
        # Gaussian blur removes camera noise before thresholding
        denoised = denoise(gray)

        # Step 4: Increase contrast
        enhanced = enhance_contrast(denoised)

        # Step 5: Adaptive threshold
        # Makes text pure black, background pure white
        thresh = adaptive_threshold(enhanced)

        # Step 6: Deskew (straighten tilted plates)
        deskewed = deskew(thresh)

        return deskewed

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        # Return grayscale as fallback
        return to_grayscale(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # already grayscale


def resize_if_small(
    image: np.ndarray,
    min_width: int = 100
) -> np.ndarray:
    """
    Upscale small images.
    Tiny plates give poor OCR — upscale with INTER_CUBIC for best quality.
    """
    h, w = image.shape[:2]

    if w < min_width:
        scale  = min_width / w
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        image  = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug(f"Resized plate from {w}x{h} to {new_w}x{new_h}")

    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove Gaussian noise using a small blur kernel.
    (3,3) kernel = mild blur, enough to smooth noise without losing text.
    """
    return cv2.GaussianBlur(image, (3, 3), 0)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Unlike global histogram equalization, CLAHE works on small regions
    so it handles uneven lighting (shadow on one side of the plate).

    clipLimit=2.0   : controls contrast amplification limit
    tileGridSize    : size of local regions
    """
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize the image.
    Adaptive = calculates threshold locally for each region.
    This handles plates with uneven lighting much better than
    a single global threshold.

    blockSize=11 : neighborhood size for threshold calculation
    C=2          : constant subtracted from mean
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Straighten slightly tilted plate images.
    Uses image moments to calculate the skew angle,
    then applies a rotation transform to correct it.

    If the plate is too tilted (>15°), we skip correction
    to avoid making things worse.
    """
    try:
        # Find all white pixels (text on white background)
        coords = np.column_stack(np.where(image > 0))

        if len(coords) < 10:
            return image  # not enough pixels to calculate skew

        # Calculate minimum bounding rectangle angle
        angle = cv2.minAreaRect(coords)[-1]

        # Normalize angle to -45 to +45 range
        if angle < -45:
            angle = 90 + angle

        # Skip if tilt is minimal or extreme
        if abs(angle) < 0.5 or abs(angle) > 15:
            return image

        # Rotate image to correct skew
        h, w   = image.shape[:2]
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags       = cv2.INTER_CUBIC,
            borderMode  = cv2.BORDER_REPLICATE
        )
        return rotated

    except Exception as e:
        logger.debug(f"Deskew skipped: {e}")
        return image


def get_multiple_preprocessed(image: np.ndarray) -> list[np.ndarray]:
    """
    Returns several differently-preprocessed versions of the same plate.
    We run OCR on all of them and pick the best result.
    This dramatically improves accuracy on difficult plates.

    Returns list of processed images.
    """
    versions = []

    gray = to_grayscale(image)
    gray = resize_if_small(gray)

    # Version 1: Standard pipeline
    versions.append(preprocess_plate(image))

    # Version 2: Inverted (white text on black — some plates)
    inverted = cv2.bitwise_not(adaptive_threshold(enhance_contrast(gray)))
    versions.append(inverted)

    # Version 3: Just grayscale + resize (sometimes simplest works best)
    versions.append(gray)

    # Version 4: Strong sharpening kernel
    sharpen_kernel = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    versions.append(sharpened)

    return versions