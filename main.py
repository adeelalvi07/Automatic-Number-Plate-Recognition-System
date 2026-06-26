# ============================================================
# FILE: main.py
# PURPOSE: Entry point — interactive menu to run any mode
# EXPLANATION:
#   Run this file to start the ANPR system.
#   It shows a menu: image / video / webcam / dashboard / train
#   Calls the appropriate detection module based on choice.
# ============================================================

import os
import sys
import logging
from pathlib import Path

# ── Setup logging ──────────────────────────────────────────
# Creates log file + shows in console simultaneously
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            log_dir / f"session_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the startup banner."""
    print("""
╔══════════════════════════════════════════════════════╗
║   🇵🇰  Pakistani Smart ANPR System                   ║
║       Automatic Number Plate Recognition             ║
║       YOLOv8 + EasyOCR + SQLite + Streamlit          ║
╚══════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print the main menu."""
    print("""
Select Mode:
─────────────────────────────────────────
  [1]  Detect on IMAGE file
  [2]  Detect on VIDEO file
  [3]  Live WEBCAM detection
  [4]  Launch Streamlit DASHBOARD
  [5]  Train YOLOv8 model
  [6]  Validate trained model
  [0]  Exit
─────────────────────────────────────────
""")


def setup_directories():
    """Create all project folders at startup."""
    from utils.save_results import setup_all_directories
    setup_all_directories()


def run_image_mode():
    """Ask for image path and run detection."""
    path = input("  Enter image path (e.g. dataset/test/images/img.jpg): ").strip()

    if not path:
        print("  No path entered.")
        return

    if not Path(path).exists():
        print(f"  File not found: {path}")
        return

    from detection.detect_image import detect_on_image
    detect_on_image(path, show_window=True, save_output=True)


def run_video_mode():
    """Ask for video path and run detection."""
    path = input("  Enter video path (e.g. test.mp4): ").strip()

    if not path:
        print("  No path entered.")
        return

    if not Path(path).exists():
        print(f"  File not found: {path}")
        return

    skip = input("  Skip every N frames for speed (default=5): ").strip()
    skip = int(skip) if skip.isdigit() else 5

    from detection.detect_video import detect_on_video
    detect_on_video(path, skip_frames=skip)


def run_webcam_mode():
    """Start live webcam detection."""
    cam = input("  Camera index (default=0): ").strip()
    cam = int(cam) if cam.isdigit() else 0

    from detection.detect_webcam import detect_on_webcam
    detect_on_webcam(camera_index=cam)


def run_dashboard():
    """Launch Streamlit dashboard in browser."""
    print("\n  Launching dashboard at http://localhost:8501")
    print("  Press Ctrl+C to stop.\n")
    os.system("streamlit run streamlit_app/app.py")


def run_training():
    """Start YOLOv8 training."""
    print("\n  Starting training...")
    print("  Make sure your dataset is in dataset/ folder first!\n")
    from training.train import train_model
    train_model()


def run_validation():
    """Validate trained model on test set."""
    from training.train import validate_model
    validate_model()


# ── Main loop ──────────────────────────────────────────────
def main():
    print_banner()
    setup_directories()

    while True:
        print_menu()
        choice = input("  Your choice: ").strip()

        if choice == "1":
            run_image_mode()

        elif choice == "2":
            run_video_mode()

        elif choice == "3":
            run_webcam_mode()

        elif choice == "4":
            run_dashboard()

        elif choice == "5":
            run_training()

        elif choice == "6":
            run_validation()

        elif choice == "0":
            print("\n  Goodbye! 👋\n")
            sys.exit(0)

        else:
            print("  Invalid choice. Please enter 1-6 or 0.")

        print()


if __name__ == "__main__":
    main()