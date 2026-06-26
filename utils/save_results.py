# ============================================================
# FILE: utils/save_results.py
# PURPOSE: Save screenshots, cropped plates, and log results
# FIXED: setup_all_directories() now creates YOUR folder layout:
#
#   dataset/train/images/   dataset/train/labels/
#   dataset/valid/images/   dataset/valid/labels/
#   dataset/test/images/    dataset/test/labels/
# ============================================================

import cv2
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
SAVE_CONFIG = {
    "screenshots_dir" : "screenshots",
    "outputs_dir"     : "outputs",
    "logs_dir"        : "logs",
    "save_screenshots": True,
    "save_crops"      : True,
}


def save_screenshot(
    frame       : np.ndarray,
    plate_text  : str = "unknown",
    source      : str = "webcam",
) -> str:
    """
    Save the full annotated frame as a screenshot.
    Filename format: YYYYMMDD_HHMMSS_PLATE_source.jpg
    """
    if not SAVE_CONFIG["save_screenshots"] or frame is None:
        return ""

    try:
        save_dir = Path(SAVE_CONFIG["screenshots_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_text = plate_text.replace("-", "").replace(" ", "_")
        filename  = f"{timestamp}_{safe_text}_{source}.jpg"
        filepath  = save_dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.debug(f"Screenshot saved: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save screenshot: {e}")
        return ""


def save_plate_crop(
    crop        : np.ndarray,
    plate_text  : str = "unknown",
) -> str:
    """
    Save the cropped plate image to outputs/ folder.
    """
    if not SAVE_CONFIG["save_crops"] or crop is None:
        return ""

    try:
        save_dir = Path(SAVE_CONFIG["outputs_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_text = plate_text.replace("-", "").replace(" ", "_")
        filename  = f"crop_{timestamp}_{safe_text}.jpg"
        filepath  = save_dir / filename

        cv2.imwrite(str(filepath), crop)
        logger.debug(f"Crop saved: {filepath}")
        return str(filepath)

    except Exception as e:
        logger.error(f"Failed to save crop: {e}")
        return ""


def write_log_entry(
    plate_text  : str,
    confidence  : float,
    source      : str,
    image_path  : str = "",
    is_duplicate: bool = False,
) -> None:
    """
    Append one line to today's log file.
    Log location: logs/YYYYMMDD.log
    """
    try:
        log_dir  = Path(SAVE_CONFIG["logs_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        today    = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{today}.log"

        time_str = datetime.now().strftime("%H:%M:%S")
        line     = (
            f"[{time_str}] "
            f"PLATE={plate_text:<12} | "
            f"CONF={confidence:.2f} | "
            f"SRC={source:<6} | "
            f"DUP={is_duplicate} | "
            f"IMG={image_path}\n"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)

    except Exception as e:
        logger.error(f"Failed to write log entry: {e}")


def setup_all_directories():
    """
    Create all required project directories at startup.

    FIXED folder structure matches your actual dataset layout:
      dataset/
      ├── train/
      │   ├── images/   ← your training images go here
      │   └── labels/   ← your training .txt labels go here
      ├── valid/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/
    """
    dirs = [
        # Dataset — YOUR actual structure
        "dataset/train/images",
        "dataset/train/labels",
        "dataset/valid/images",
        "dataset/valid/labels",
        "dataset/test/images",
        "dataset/test/labels",

        # Model output
        "models",

        # Detection outputs
        "outputs",
        "screenshots",

        # Database
        "database",

        # Logs
        "logs",

        # App folders (already have files, just ensure they exist)
        "streamlit_app",
        "utils",
        "detection",
        "training",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("All project directories ready.")
    logger.info("Dataset folder structure:")
    logger.info("  dataset/train/images/  ✓")
    logger.info("  dataset/train/labels/  ✓")
    logger.info("  dataset/valid/images/  ✓")
    logger.info("  dataset/valid/labels/  ✓")
    logger.info("  dataset/test/images/   ✓")
    logger.info("  dataset/test/labels/   ✓")