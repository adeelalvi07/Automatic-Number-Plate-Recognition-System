# ============================================================
# FILE: utils/inference.py
# PURPOSE: YOLOv8 detection wrapper
# EXPLANATION:
#   - Loads your trained model once (efficient)
#   - Runs detection on any frame/image
#   - Returns bounding boxes, confidence scores, cropped plates
#   - Works for webcam frames, images, and video frames
# ============================================================

import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
INFERENCE_CONFIG = {
    "model_path"      : "models/best.pt",
    "confidence"      : 0.40,      # minimum confidence to accept detection
    "iou_threshold"   : 0.45,      # overlap threshold for NMS
    "imgsz"           : 640,       # inference image size
    "device"          : "cpu",     # 'cpu' or '0' for GPU
    "pad_ratio"       : 0.05,      # extra padding around cropped plate
}


class PlateDetector:
    """
    YOLOv8-based license plate detector.

    Usage:
        detector = PlateDetector()
        results  = detector.detect(frame)
    """

    def __init__(self, model_path: str = None):
        """
        Load the YOLOv8 model.
        model_path: path to your best.pt file
        """
        self.model_path = model_path or INFERENCE_CONFIG["model_path"]
        self.model      = None
        self.conf       = INFERENCE_CONFIG["confidence"]
        self.iou        = INFERENCE_CONFIG["iou_threshold"]
        self.imgsz      = INFERENCE_CONFIG["imgsz"]
        self.device     = INFERENCE_CONFIG["device"]
        self.pad_ratio  = INFERENCE_CONFIG["pad_ratio"]

        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model from disk."""
        if not Path(self.model_path).exists():
            logger.error(f"Model not found at: {self.model_path}")
            logger.error("Please train the model first: python training/train.py")
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("Model loaded successfully!")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run plate detection on a single frame.

        Args:
            frame: BGR image as NumPy array (from OpenCV)

        Returns:
            List of dicts, each containing:
              - bbox        : [x1, y1, x2, y2] bounding box
              - confidence  : float 0-1
              - crop        : cropped plate image (NumPy array)
              - label       : class name string
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame passed to detector.")
            return []

        detections = []

        try:
            # Run YOLOv8 inference
            # verbose=False silences per-frame console output
            results = self.model.predict(
                source  = frame,
                conf    = self.conf,
                iou     = self.iou,
                imgsz   = self.imgsz,
                device  = self.device,
                verbose = False,
            )

            # Parse results
            for result in results:
                boxes = result.boxes  # YOLOv8 Boxes object

                for box in boxes:
                    # Get bounding box in pixel coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence      = float(box.conf[0])
                    class_id        = int(box.cls[0])
                    label           = result.names[class_id]

                    # Crop the plate region with padding
                    crop = self._crop_with_padding(frame, x1, y1, x2, y2)

                    detections.append({
                        "bbox"       : [x1, y1, x2, y2],
                        "confidence" : round(confidence, 4),
                        "crop"       : crop,
                        "label"      : label,
                    })

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return detections

    def _crop_with_padding(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """
        Crop the plate region from the frame with a small padding
        so we don't cut off the plate edges.

        pad_ratio: 5% extra on each side
        """
        h, w = frame.shape[:2]

        pad_x = int((x2 - x1) * self.pad_ratio)
        pad_y = int((y2 - y1) * self.pad_ratio)

        # Clamp to image boundaries so we don't go out of bounds
        x1_p = max(0, x1 - pad_x)
        y1_p = max(0, y1 - pad_y)
        x2_p = min(w, x2 + pad_x)
        y2_p = min(h, y2 + pad_y)

        return frame[y1_p:y2_p, x1_p:x2_p].copy()

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        plate_texts: list[str] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes and plate text on the frame.

        Args:
            frame       : original BGR frame
            detections  : list from self.detect()
            plate_texts : optional list of OCR results to overlay

        Returns:
            Annotated frame (BGR NumPy array)
        """
        annotated = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            conf            = det["confidence"]

            # Draw bounding box — green rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Build label text
            label = f"Plate: {conf:.2f}"
            if plate_texts and i < len(plate_texts):
                label = f"{plate_texts[i]}  ({conf:.2f})"

            # Draw label background (filled rectangle behind text)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 10),
                (x1 + tw + 6, y1),
                (0, 255, 0), -1          # -1 = filled
            )

            # Draw label text — black on green background
            cv2.putText(
                annotated, label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2
            )

        return annotated


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS counter on top-left corner of frame."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 255), 2
    )
    return frame