# 🇵🇰 Pakistani Smart ANPR System

**Automatic Number Plate Recognition** for Pakistani vehicles  
Built with **YOLOv8 · EasyOCR · OpenCV · SQLite · Streamlit**

---

## Project Structure
ANPR_Project/
├── dataset/           ← Your training dataset (YOLO format)
├── models/            ← Trained YOLOv8 weights (best.pt)
├── outputs/           ← Cropped plate images
├── screenshots/       ← Full-frame annotated screenshots
├── database/          ← SQLite database (anpr.db)
├── logs/              ← Daily log files
├── streamlit_app/
│   └── app.py         ← Web dashboard
├── utils/
│   ├── inference.py   ← YOLOv8 detection
│   ├── preprocess.py  ← Image enhancement
│   ├── ocr_reader.py  ← EasyOCR wrapper
│   ├── text_cleaner.py← Pakistani plate validation
│   ├── database.py    ← SQLite operations
│   └── save_results.py← File saving utilities
├── detection/
│   ├── detect_image.py
│   ├── detect_video.py
│   └── detect_webcam.py
├── training/
│   └── train.py
├── main.py
├── requirements.txt
└── README.md