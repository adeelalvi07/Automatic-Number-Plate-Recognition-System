# ============================================================
# FILE: detection/detect_video.py
# PURPOSE: Run ANPR on a video file (MP4/AVI etc.)
# EXPLANATION:
#   Opens a video file frame-by-frame.
#   Runs detection every N frames (skip_frames) for speed.
#   Calculates and displays FPS in real-time.
#   Saves results to DB, skipping duplicates.
# ============================================================

import cv2
import time
import logging
from pathlib import Path

from utils.inference    import PlateDetector, draw_fps
from utils.ocr_reader   import read_plate
from utils.database     import insert_detection, init_database
from utils.save_results import save_screenshot, save_plate_crop, write_log_entry

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
VIDEO_CONFIG = {
    "skip_frames"   : 5,       # run detection every N frames (speed vs accuracy)
    "show_window"   : True,    # display video window
    "save_output"   : True,    # save screenshots
    "window_name"   : "ANPR - Video Detection",
    "exit_key"      : ord("q"),  # press Q to quit
}


def detect_on_video(
    video_path   : str,
    skip_frames  : int  = VIDEO_CONFIG["skip_frames"],
    show_window  : bool = VIDEO_CONFIG["show_window"],
    save_output  : bool = VIDEO_CONFIG["save_output"],
) -> list[dict]:
    """
    Run full ANPR pipeline on a video file.

    Args:
        video_path  : path to the video file
        skip_frames : process every Nth frame
        show_window : show live annotated video
        save_output : save screenshots on new detections

    Returns:
        All unique detections across the video.
    """
    logger.info(f"Processing video: {video_path}")

    # ── Validate ────────────────────────────────────────────
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        return []

    # ── Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps_video:.1f} FPS")

    # ── Initialize ──────────────────────────────────────────
    init_database()
    detector = PlateDetector()

    all_results  = []
    frame_count  = 0
    prev_time    = time.time()
    fps_display  = 0.0

    # ── Frame loop ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break   # end of video

        frame_count += 1

        # ── FPS calculation ─────────────────────────────────
        curr_time   = time.time()
        fps_display = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time   = curr_time

        # ── Skip frames for performance ──────────────────────
        # Detection is expensive — run every N frames only
        if frame_count % skip_frames != 0:
            # Still show the frame (with previous annotations)
            if show_window:
                display = draw_fps(frame.copy(), fps_display)
                cv2.imshow(VIDEO_CONFIG["window_name"], display)
                if cv2.waitKey(1) & 0xFF == VIDEO_CONFIG["exit_key"]:
                    logger.info("User pressed Q — stopping.")
                    break
            continue

        # ── Run detection ────────────────────────────────────
        detections  = detector.detect(frame)
        plate_texts = []

        for det in detections:
            ocr_result = read_plate(det["crop"])
            plate_text = ocr_result["text"]

            if not plate_text:
                plate_texts.append("?")
                continue

            plate_texts.append(plate_text)

            # Save crop
            crop_path = ""
            if save_output:
                crop_path = save_plate_crop(det["crop"], plate_text)

            # DB insert
            record_id, is_dup = insert_detection(
                plate_text   = plate_text,
                confidence   = ocr_result["confidence"],
                image_path   = crop_path,
                source       = "video",
                vehicle_conf = det["confidence"],
            )

            write_log_entry(
                plate_text,
                ocr_result["confidence"],
                "video",
                crop_path,
                is_dup,
            )

            if not is_dup:
                # Save screenshot on new unique detection
                if save_output:
                    annotated_snap = detector.draw_detections(
                        frame, detections, plate_texts
                    )
                    save_screenshot(annotated_snap, plate_text, "video")

                all_results.append({
                    "plate_text" : plate_text,
                    "confidence" : ocr_result["confidence"],
                    "frame"      : frame_count,
                    "record_id"  : record_id,
                })

                logger.info(
                    f"Frame {frame_count}/{total_frames} | "
                    f"Plate: {plate_text} | "
                    f"Conf: {ocr_result['confidence']:.2f}"
                )

        # ── Annotate and show ────────────────────────────────
        if show_window:
            annotated = detector.draw_detections(frame, detections, plate_texts)
            annotated = draw_fps(annotated, fps_display)

            # Progress bar overlay
            progress = frame_count / max(total_frames, 1)
            bar_w    = int(annotated.shape[1] * progress)
            cv2.rectangle(annotated, (0, 0), (bar_w, 5), (0, 255, 0), -1)

            cv2.imshow(VIDEO_CONFIG["window_name"], annotated)
            if cv2.waitKey(1) & 0xFF == VIDEO_CONFIG["exit_key"]:
                logger.info("User pressed Q — stopping.")
                break

    # ── Cleanup ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    logger.info(f"Video processing complete. {len(all_results)} unique plates found.")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  VIDEO PROCESSING COMPLETE")
    print(f"  Frames processed : {frame_count}")
    print(f"  Unique plates    : {len(all_results)}")
    print("="*50)
    for r in all_results:
        print(f"  [{r['frame']:5d}] {r['plate_text']} ({r['confidence']:.2f})")

    return all_results