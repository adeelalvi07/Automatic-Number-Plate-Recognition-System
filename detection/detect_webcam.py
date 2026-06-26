# ============================================================
# FILE: detection/detect_webcam.py
# PURPOSE: Run ANPR on live webcam feed
# EXPLANATION:
#   Opens webcam (camera index 0) in a loop.
#   Runs detection every N frames for real-time performance.
#   Press Q to quit, S to save a manual screenshot.
#   Shows FPS counter and bounding box overlays live.
# ============================================================

import cv2
import time
import logging

from utils.inference    import PlateDetector, draw_fps
from utils.ocr_reader   import read_plate
from utils.database     import insert_detection, init_database
from utils.save_results import save_screenshot, save_plate_crop, write_log_entry

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
WEBCAM_CONFIG = {
    "camera_index"  : 0,       # 0 = default webcam, 1 = external
    "frame_width"   : 1280,    # capture resolution width
    "frame_height"  : 720,     # capture resolution height
    "skip_frames"   : 3,       # detect every N frames
    "window_name"   : "ANPR - Live Webcam  |  Q=Quit  S=Screenshot",
    "exit_key"      : ord("q"),
    "screenshot_key": ord("s"),
}


def detect_on_webcam(
    camera_index : int  = WEBCAM_CONFIG["camera_index"],
    skip_frames  : int  = WEBCAM_CONFIG["skip_frames"],
) -> None:
    """
    Main webcam detection loop.
    Runs indefinitely until user presses Q.

    Args:
        camera_index : which camera to open (0 = built-in)
        skip_frames  : detect every N frames
    """
    logger.info(f"Starting webcam (camera index: {camera_index})")

    # ── Open webcam ─────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        logger.error(
            f"Cannot open camera {camera_index}. "
            "Check if your webcam is connected."
        )
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_CONFIG["frame_height"])

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Webcam opened: {actual_w}x{actual_h}")

    # ── Initialize ──────────────────────────────────────────
    init_database()
    detector = PlateDetector()

    frame_count = 0
    prev_time   = time.time()
    fps_display = 0.0

    # Track last shown detections (for display between detection frames)
    last_detections  = []
    last_plate_texts = []

    # ── Main loop ────────────────────────────────────────────
    logger.info("Webcam running. Press Q to quit, S to screenshot.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from webcam.")
            continue

        frame_count += 1

        # ── FPS ─────────────────────────────────────────────
        curr_time   = time.time()
        fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time   = curr_time

        # ── Run detection every N frames ────────────────────
        if frame_count % skip_frames == 0:
            last_detections  = detector.detect(frame)
            last_plate_texts = []

            for det in last_detections:
                ocr_result = read_plate(det["crop"])
                plate_text = ocr_result["text"]

                if not plate_text:
                    last_plate_texts.append("?")
                    continue

                last_plate_texts.append(plate_text)

                # Save crop
                crop_path = save_plate_crop(det["crop"], plate_text)

                # DB insert (handles duplicates internally)
                record_id, is_dup = insert_detection(
                    plate_text   = plate_text,
                    confidence   = ocr_result["confidence"],
                    image_path   = crop_path,
                    source       = "webcam",
                    vehicle_conf = det["confidence"],
                )

                write_log_entry(
                    plate_text,
                    ocr_result["confidence"],
                    "webcam",
                    crop_path,
                    is_dup,
                )

                if not is_dup:
                    logger.info(
                        f"NEW PLATE: {plate_text} | "
                        f"Conf: {ocr_result['confidence']:.2f}"
                    )

        # ── Draw annotations on every frame ─────────────────
        # (using last known detections for smooth display)
        annotated = detector.draw_detections(
            frame,
            last_detections,
            last_plate_texts
        )
        annotated = draw_fps(annotated, fps_display)

        # ── Instructions overlay ─────────────────────────────
        cv2.putText(
            annotated,
            "Q=Quit  S=Screenshot",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1
        )

        # ── Show window ─────────────────────────────────────
        cv2.imshow(WEBCAM_CONFIG["window_name"], annotated)

        # ── Key handling ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == WEBCAM_CONFIG["exit_key"]:
            logger.info("User pressed Q — stopping webcam.")
            break

        elif key == WEBCAM_CONFIG["screenshot_key"]:
            plate = last_plate_texts[0] if last_plate_texts else "manual"
            path  = save_screenshot(annotated, plate, "webcam")
            logger.info(f"Manual screenshot saved: {path}")
            # Flash a white overlay to signal save
            flash = annotated.copy()
            flash[:] = (255, 255, 255)
            cv2.addWeighted(flash, 0.3, annotated, 0.7, 0, annotated)
            cv2.imshow(WEBCAM_CONFIG["window_name"], annotated)
            cv2.waitKey(100)

    # ── Cleanup ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam session ended.")