# ============================================================
# FILE: detection/detect_image.py
# PURPOSE: Run ANPR on a single image file
# EXPLANATION:
#   Loads an image → detects plates → reads OCR → saves results.
#   Returns a list of all detections so main.py can display them.
# ============================================================

import cv2
import logging
from pathlib import Path

from utils.inference   import PlateDetector
from utils.ocr_reader  import read_plate
from utils.database    import insert_detection, init_database
from utils.save_results import save_screenshot, save_plate_crop, write_log_entry

logger = logging.getLogger(__name__)


def detect_on_image(
    image_path   : str,
    show_window  : bool = True,
    save_output  : bool = True,
) -> list[dict]:
    """
    Run full ANPR pipeline on a single image.

    Args:
        image_path  : path to input image (JPG/PNG)
        show_window : display annotated image in a window
        save_output : save screenshot and crops

    Returns:
        List of detection dicts:
          [{ plate_text, confidence, bbox, image_path, ... }, ...]
    """
    logger.info(f"Processing image: {image_path}")

    # ── Validate input ──────────────────────────────────────
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return []

    # ── Load image ─────────────────────────────────────────
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Could not read image: {image_path}")
        return []

    logger.info(f"Image loaded: {frame.shape[1]}x{frame.shape[0]} px")

    # ── Initialize ──────────────────────────────────────────
    init_database()
    detector = PlateDetector()

    # ── Detection ───────────────────────────────────────────
    detections = detector.detect(frame)
    logger.info(f"Found {len(detections)} plate(s)")

    if not detections:
        logger.info("No plates detected in this image.")
        if show_window:
            cv2.imshow("ANPR - No Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return []

    # ── OCR on each detection ───────────────────────────────
    results       = []
    plate_texts   = []

    for i, det in enumerate(detections):
        logger.info(f"Running OCR on plate {i+1}/{len(detections)}...")

        ocr_result   = read_plate(det["crop"])
        plate_text   = ocr_result["text"] or f"PLATE_{i+1}"
        ocr_conf     = ocr_result["confidence"]
        is_valid     = ocr_result["valid"]

        plate_texts.append(plate_text)

        # ── Save crop ───────────────────────────────────────
        crop_path = ""
        if save_output:
            crop_path = save_plate_crop(det["crop"], plate_text)

        # ── Save to DB ──────────────────────────────────────
        record_id, is_dup = insert_detection(
            plate_text   = plate_text,
            confidence   = ocr_conf,
            image_path   = crop_path,
            source       = "image",
            vehicle_conf = det["confidence"],
        )

        # ── Log entry ───────────────────────────────────────
        write_log_entry(plate_text, ocr_conf, "image", crop_path, is_dup)

        result = {
            "plate_text"  : plate_text,
            "confidence"  : ocr_conf,
            "valid"       : is_valid,
            "bbox"        : det["bbox"],
            "yolo_conf"   : det["confidence"],
            "record_id"   : record_id,
            "is_duplicate": is_dup,
            "image_path"  : crop_path,
        }
        results.append(result)

        logger.info(
            f"  Plate: {plate_text} | "
            f"OCR conf: {ocr_conf:.2f} | "
            f"Valid: {is_valid} | "
            f"Duplicate: {is_dup}"
        )

    # ── Annotate frame ──────────────────────────────────────
    annotated = detector.draw_detections(frame, detections, plate_texts)

    # ── Save screenshot ─────────────────────────────────────
    if save_output:
        main_plate = plate_texts[0] if plate_texts else "unknown"
        save_screenshot(annotated, main_plate, "image")

    # ── Show window ─────────────────────────────────────────
    if show_window:
        # Resize if image is too large for screen
        h, w = annotated.shape[:2]
        if w > 1280:
            scale    = 1280 / w
            annotated = cv2.resize(
                annotated,
                (int(w * scale), int(h * scale))
            )

        cv2.imshow("ANPR - Image Detection", annotated)
        logger.info("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Print summary ───────────────────────────────────────
    print("\n" + "="*50)
    print(f"  DETECTIONS FOUND: {len(results)}")
    print("="*50)
    for r in results:
        print(f"  Plate : {r['plate_text']}")
        print(f"  Conf  : {r['confidence']:.2f}")
        print(f"  Valid : {r['valid']}")
        print("-"*50)

    return results