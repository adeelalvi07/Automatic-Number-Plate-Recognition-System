# ============================================================
# FILE: training/train.py
# PURPOSE: Train YOLOv8 on your Pakistani license plate dataset
# FIXED: Dataset paths updated to match actual folder structure:
#
#   YOUR ACTUAL STRUCTURE:
#   dataset/
#   ├── train/
#   │   ├── images/
#   │   └── labels/
#   ├── valid/
#   │   ├── images/
#   │   └── labels/
#   └── test/
#       ├── images/
#       └── labels/
# ============================================================

import os
import yaml
import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── Setup logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────
TRAIN_CONFIG = {
    "model_size"  : "yolov8n.pt",   # n=nano, s=small, m=medium
    "epochs"      : 30,
    "imgsz"       : 640,
    "batch"       : 16,              # lower to 8 if you get memory errors
    "device"      : "0",           # 'cpu' or '0' for GPU
    "patience"    : 10,
    "workers"     : 0,
    "project"     : "models",
    "name"        : "plate_detector",
    "exist_ok"    : True,
}

DATASET_YAML = "dataset/dataset.yaml"


def create_dataset_yaml():
    """
    Creates dataset.yaml for YOLOv8.

    YOUR folder layout:
      dataset/train/images/   ← training images
      dataset/train/labels/   ← training labels (.txt)
      dataset/valid/images/   ← validation images
      dataset/valid/labels/   ← validation labels
      dataset/test/images/    ← test images
      dataset/test/labels/    ← test labels

    YOLOv8 needs absolute paths to avoid confusion
    when running from different working directories.
    """
    # Get absolute path to dataset folder
    dataset_root = Path("dataset").absolute()

    # Verify folders exist before writing yaml
    required_folders = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
        dataset_root / "test"  / "images",
        dataset_root / "test"  / "labels",
    ]

    missing = [str(f) for f in required_folders if not f.exists()]
    if missing:
        logger.warning("These dataset folders are missing:")
        for m in missing:
            logger.warning(f"  ✗ {m}")
        logger.warning("Training will proceed but may fail if images are missing.")
    else:
        logger.info("All dataset folders found ✓")

    # Count images in each split
    for split in ["train", "valid", "test"]:
        img_dir = dataset_root / split / "images"
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")) +
                        list(img_dir.glob("*.jpeg")) +
                        list(img_dir.glob("*.png")))
            logger.info(f"  {split}: {count} images found")

    # Build the YAML config
    # Note: paths are relative to the 'path' key (dataset root)
    config = {
        "path"  : str(dataset_root),   # absolute root
        "train" : "train/images",       # relative to path
        "val"   : "valid/images",       # relative to path
        "test"  : "test/images",        # relative to path
        "nc"    : 1,                    # number of classes
        "names" : ["license_plate"],    # class names list
    }

    # Save the yaml file inside dataset/
    yaml_path = Path(DATASET_YAML)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"dataset.yaml created at: {yaml_path.absolute()}")
    logger.info(f"Contents:\n  path : {config['path']}")
    logger.info(f"  train: {config['train']}")
    logger.info(f"  val  : {config['val']}")
    logger.info(f"  test : {config['test']}")

    return str(yaml_path)


def train_model():
    """
    Main training function.
    1. Creates dataset.yaml with correct paths
    2. Downloads YOLOv8 nano pretrained weights
    3. Fine-tunes on your plate dataset
    4. Saves best.pt to models/best.pt
    """
    logger.info("=" * 55)
    logger.info("  Pakistani ANPR — YOLOv8 Training Started")
    logger.info("=" * 55)

    # Step 1: Create dataset config
    yaml_path = create_dataset_yaml()

    # Step 2: Load YOLOv8 base model
    logger.info(f"Loading base model: {TRAIN_CONFIG['model_size']}")
    model = YOLO(TRAIN_CONFIG["model_size"])

    # Step 3: Train
    logger.info(f"Training for {TRAIN_CONFIG['epochs']} epochs...")
    logger.info(f"Using device: {TRAIN_CONFIG['device']}")
    logger.info("This will take a while on CPU. GPU is much faster.")

    results = model.train(
        data      = yaml_path,
        epochs    = TRAIN_CONFIG["epochs"],
        imgsz     = TRAIN_CONFIG["imgsz"],
        batch     = TRAIN_CONFIG["batch"],
        device    = TRAIN_CONFIG["device"],
        patience  = TRAIN_CONFIG["patience"],
        workers   = TRAIN_CONFIG["workers"],
        project   = TRAIN_CONFIG["project"],
        name      = TRAIN_CONFIG["name"],
        exist_ok  = TRAIN_CONFIG["exist_ok"],
        verbose   = True,
    )

    # Step 4: Copy best weights to models/best.pt
    best_src = Path(f"models/{TRAIN_CONFIG['name']}/weights/best.pt")
    best_dst = Path("models/best.pt")

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        logger.info(f"Best model copied to: {best_dst.absolute()}")
    else:
        logger.warning(
            f"best.pt not found at {best_src}. "
            "Check models/plate_detector/weights/ manually."
        )

    logger.info("=" * 55)
    logger.info("  Training Complete!")
    logger.info("=" * 55)
    return results


def validate_model():
    """
    Validate the trained model on the test split.
    Reports mAP50 and mAP50-95 (standard detection metrics).

    mAP50     = Mean Average Precision at IoU threshold 0.50
    mAP50-95  = averaged across IoU thresholds 0.50 to 0.95
    Higher is better. Good model: mAP50 > 0.85
    """
    model_path = Path("models/best.pt")

    if not model_path.exists():
        logger.error(
            "models/best.pt not found. "
            "Please run training first."
        )
        return None

    logger.info("Running validation on test set...")
    model   = YOLO(str(model_path))
    metrics = model.val(
        data  = DATASET_YAML,
        split = "test",
    )

    logger.info("─" * 40)
    logger.info(f"  mAP50     : {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95  : {metrics.box.map:.4f}")
    logger.info(f"  Precision : {metrics.box.mp:.4f}")
    logger.info(f"  Recall    : {metrics.box.mr:.4f}")
    logger.info("─" * 40)

    return metrics


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure all required folders exist
    folders_to_create = [
        "models",
        "dataset/train/images",
        "dataset/train/labels",
        "dataset/valid/images",
        "dataset/valid/labels",
        "dataset/test/images",
        "dataset/test/labels",
        "logs",
    ]
    for folder in folders_to_create:
        Path(folder).mkdir(parents=True, exist_ok=True)

    train_model()
    validate_model()