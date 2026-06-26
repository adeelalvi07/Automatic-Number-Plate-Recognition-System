# ============================================================
# FILE: utils/ocr_reader.py
# PURPOSE: Read text from plate images using EasyOCR
# EXPLANATION:
#   EasyOCR is a deep learning OCR engine.
#   It supports multiple languages including English + Urdu.
#   We load it ONCE (heavy model) and reuse it for every frame.
#   We try multiple preprocessed versions and pick the best result.
# ============================================================

import numpy as np
import logging
import easyocr
from utils.preprocess import get_multiple_preprocessed
from utils.text_cleaner import clean_plate_text, is_valid_pakistani_plate

logger = logging.getLogger(__name__)


# ── Singleton OCR Reader ───────────────────────────────────
# EasyOCR takes ~3-5 seconds to load — we only do it once
_reader_instance = None

def get_ocr_reader() -> easyocr.Reader:
    """
    Returns a singleton EasyOCR reader.
    Languages: English (all plates) — can add 'ur' for Urdu if needed.
    gpu=False  : use CPU (set True if you have CUDA GPU)
    """
    global _reader_instance
    if _reader_instance is None:
        logger.info("Loading EasyOCR model (first load may take 30 seconds)...")
        _reader_instance = easyocr.Reader(
            ["en"],          # language list
            gpu=False,       # set True for GPU acceleration
            verbose=False,   # suppress EasyOCR internal logs
        )
        logger.info("EasyOCR loaded successfully!")
    return _reader_instance


# ── Main OCR Function ──────────────────────────────────────

def read_plate(crop: np.ndarray) -> dict:
    """
    Extract text from a cropped license plate image.

    Strategy:
      1. Generate multiple preprocessed versions of the plate crop
      2. Run EasyOCR on each version
      3. Pick the result with highest confidence
      4. Clean and validate the text

    Args:
        crop: BGR/grayscale NumPy array of the plate region

    Returns:
        dict with keys:
          - text       : cleaned plate text (e.g. "ABC-123")
          - raw_text   : unprocessed OCR output
          - confidence : float 0-1
          - valid      : bool (matches Pakistani format)
    """
    if crop is None or crop.size == 0:
        return _empty_result()

    reader = get_ocr_reader()

    # Get multiple preprocessed versions to try
    versions = get_multiple_preprocessed(crop)

    best_result = _empty_result()

    for version in versions:
        try:
            result = _run_ocr_on_image(reader, version)

            # Keep the result with highest confidence
            if result["confidence"] > best_result["confidence"]:
                best_result = result

            # If we already got a valid plate with high confidence, stop
            if best_result["valid"] and best_result["confidence"] > 0.75:
                break

        except Exception as e:
            logger.debug(f"OCR attempt failed: {e}")
            continue

    logger.debug(
        f"OCR result: '{best_result['text']}' "
        f"(conf={best_result['confidence']:.2f}, "
        f"valid={best_result['valid']})"
    )

    return best_result


def _run_ocr_on_image(
    reader: easyocr.Reader,
    image: np.ndarray
) -> dict:
    """
    Run EasyOCR on a single preprocessed image.

    detail=1         : return bounding boxes + confidence
    paragraph=False  : treat each text region separately
    allowlist        : only recognize these characters (reduces errors)
                       Pakistani plates use A-Z and 0-9
    """
    # Character allowlist: uppercase letters + digits + hyphen/space
    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- "

    results = reader.readtext(
        image,
        detail       = 1,
        paragraph    = False,
        allowlist    = allowlist,
        batch_size   = 1,
        text_threshold   = 0.5,   # minimum text confidence
        link_threshold   = 0.4,
        low_text         = 0.4,
    )

    if not results:
        return _empty_result()

    # Combine all detected text regions
    # (a plate might be read as multiple segments)
    all_text   = " ".join([r[1] for r in results])
    avg_conf   = sum([r[2] for r in results]) / len(results)

    # Clean the raw text
    cleaned    = clean_plate_text(all_text)
    is_valid   = is_valid_pakistani_plate(cleaned)

    return {
        "text"       : cleaned,
        "raw_text"   : all_text,
        "confidence" : round(avg_conf, 4),
        "valid"      : is_valid,
    }


def _empty_result() -> dict:
    """Return an empty result dict for failed OCR."""
    return {
        "text"       : "",
        "raw_text"   : "",
        "confidence" : 0.0,
        "valid"      : False,
    }