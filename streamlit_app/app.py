# ============================================================
# FILE: streamlit_app/app.py
# PURPOSE: Full Streamlit dashboard with:
#   - Image upload + detection
#   - Video upload + detection
#   - Live webcam detection
#   - Detection history
#   - Charts and statistics
#   - CSV export
# ============================================================

import sys
import os
import cv2
import time
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from utils.database import (
    init_database,
    get_all_detections,
    get_unique_plates,
    get_stats,
    export_to_csv,
    delete_all_records,
)
from utils.inference    import PlateDetector, draw_fps
from utils.ocr_reader   import read_plate
from utils.save_results import save_screenshot, save_plate_crop, write_log_entry
from utils.database     import insert_detection

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title = "Pakistani ANPR Dashboard",
    page_icon  = "🚗",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Initialize database ────────────────────────────────────
init_database()

# ── Load detector once (cached so it doesn't reload) ──────
@st.cache_resource
def load_detector():
    """
    Load YOLOv8 model once and cache it.
    st.cache_resource keeps it in memory across reruns.
    """
    try:
        detector = PlateDetector()
        return detector
    except Exception as e:
        st.error(f"Model not found: {e}")
        st.error("Please train the model first: python training/train.py")
        return None

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        background: #1e3a1e;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .plate-text {
        font-size: 32px;
        font-weight: bold;
        color: #28a745;
        font-family: monospace;
        letter-spacing: 3px;
    }
    .conf-text {
        font-size: 14px;
        color: #aaaaaa;
    }
    .valid-badge {
        background: #28a745;
        color: white;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 12px;
    }
    .invalid-badge {
        background: #dc3545;
        color: white;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🇵🇰 Pakistani ANPR")
    st.caption("Smart Number Plate Recognition")
    st.divider()

    # Navigation menu
    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home & Stats",
            "🖼️ Image Detection",
            "🎬 Video Detection",
            "📷 Webcam Detection",
            "📋 Detection History",
            "📊 Analytics",
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # Quick stats in sidebar
    stats = get_stats()
    st.metric("Total Detections", stats["total_detections"])
    st.metric("Unique Plates",    stats["unique_plates"])
    st.metric("Today",            stats["today_detections"])

    st.divider()

    # Export button
    if st.button("📥 Export CSV", use_container_width=True):
        csv_path = export_to_csv()
        df_exp   = get_all_detections(limit=100000)
        csv_data = df_exp.to_csv(index=False)
        st.download_button(
            label     = "⬇️ Download Now",
            data      = csv_data,
            file_name = f"anpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime      = "text/csv",
        )

    # Danger zone
    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Delete ALL records", type="secondary"):
            delete_all_records()
            st.warning("All records deleted!")
            st.rerun()


# ════════════════════════════════════════════════════════════
# HELPER FUNCTION — Run detection on a single frame
# ════════════════════════════════════════════════════════════
def run_detection_on_frame(frame, detector, source="image"):
    """
    Run full ANPR pipeline on one frame.
    Returns annotated frame and list of results.
    """
    results      = []
    plate_texts  = []

    detections = detector.detect(frame)

    for det in detections:
        ocr_result = read_plate(det["crop"])
        plate_text = ocr_result["text"] or "UNKNOWN"
        plate_texts.append(plate_text)

        # Save crop
        crop_path = save_plate_crop(det["crop"], plate_text)

        # Save to database
        record_id, is_dup = insert_detection(
            plate_text   = plate_text,
            confidence   = ocr_result["confidence"],
            image_path   = crop_path,
            source       = source,
            vehicle_conf = det["confidence"],
        )

        write_log_entry(
            plate_text,
            ocr_result["confidence"],
            source,
            crop_path,
            is_dup,
        )

        results.append({
            "plate_text"  : plate_text,
            "confidence"  : ocr_result["confidence"],
            "valid"       : ocr_result["valid"],
            "bbox"        : det["bbox"],
            "yolo_conf"   : det["confidence"],
            "is_duplicate": is_dup,
            "crop"        : det["crop"],
        })

    # Draw annotations
    annotated = detector.draw_detections(frame, detections, plate_texts)
    return annotated, results


def show_detection_results(results):
    """Display detection results in nice cards."""
    if not results:
        st.warning("No license plates detected in this input.")
        return

    st.success(f"✅ Found {len(results)} plate(s)!")

    for r in results:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-box">
                <div class="plate-text">{r['plate_text']}</div>
                <div class="conf-text">License Plate</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("OCR Confidence",  f"{r['confidence']:.1%}")
            st.metric("YOLO Confidence", f"{r['yolo_conf']:.1%}")

        with col3:
            if r["valid"]:
                st.markdown('<span class="valid-badge">✅ Valid Pakistani Plate</span>',
                           unsafe_allow_html=True)
            else:
                st.markdown('<span class="invalid-badge">⚠️ Format Unknown</span>',
                           unsafe_allow_html=True)

            if r["is_duplicate"]:
                st.caption("🔄 Duplicate (seen recently)")
            else:
                st.caption("🆕 New detection saved")


# ════════════════════════════════════════════════════════════
# PAGE 1 — HOME & STATS
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Stats":
    st.title("🇵🇰 Pakistani Smart ANPR System")
    st.caption("Automatic Number Plate Recognition powered by YOLOv8 + EasyOCR")
    st.divider()

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    stats = get_stats()

    with col1:
        st.metric("📸 Total Detections", stats["total_detections"])
    with col2:
        st.metric("🚗 Unique Plates",    stats["unique_plates"])
    with col3:
        st.metric("📅 Today",            stats["today_detections"])
    with col4:
        model_exists = Path("models/best.pt").exists()
        st.metric("🤖 Model Status", "✅ Ready" if model_exists else "❌ Not trained")

    st.divider()

    # How to use guide
    st.subheader("🚀 How to Use")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("""
        **🖼️ Image Detection**
        Upload any JPG or PNG image
        containing a vehicle or
        license plate.
        """)
    with c2:
        st.info("""
        **🎬 Video Detection**
        Upload an MP4 or AVI video.
        System processes every
        5th frame automatically.
        """)
    with c3:
        st.info("""
        **📷 Webcam Detection**
        Use your PC camera live.
        Hold a plate in front and
        click Capture to detect.
        """)

    st.divider()

    # Recent detections preview
    st.subheader("🕐 Recent Detections")
    df = get_all_detections(limit=5)
    if not df.empty:
        st.dataframe(
            df[["plate_text", "confidence", "timestamp", "source"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No detections yet. Use Image/Video/Webcam tabs to start detecting.")


# ════════════════════════════════════════════════════════════
# PAGE 2 — IMAGE DETECTION
# ════════════════════════════════════════════════════════════
elif page == "🖼️ Image Detection":
    st.title("🖼️ Image Detection")
    st.caption("Upload an image to detect and read license plates")
    st.divider()

    # Load model
    detector = load_detector()
    if detector is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image",
        type        = ["jpg", "jpeg", "png", "bmp"],
        help        = "Supported formats: JPG, JPEG, PNG, BMP"
    )

    if uploaded_file is not None:
        # Show original image
        col1, col2 = st.columns(2)

        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8
        )
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with col1:
            st.subheader("Original Image")
            # Convert BGR to RGB for display
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_column_width=True)

        # Run detection button
        if st.button("🔍 Detect License Plates", type="primary", use_container_width=True):
            with st.spinner("Running detection..."):
                annotated, results = run_detection_on_frame(
                    frame, detector, source="image"
                )

            with col2:
                st.subheader("Detection Result")
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)

                # Save screenshot
                save_screenshot(annotated, 
                    results[0]["plate_text"] if results else "unknown",
                    "image"
                )

            st.divider()
            show_detection_results(results)

            # Show cropped plates
            if results:
                st.subheader("🔍 Cropped Plates")
                crop_cols = st.columns(len(results))
                for i, r in enumerate(results):
                    with crop_cols[i]:
                        crop_rgb = cv2.cvtColor(r["crop"], cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, caption=r["plate_text"], width=200)


# ════════════════════════════════════════════════════════════
# PAGE 3 — VIDEO DETECTION
# ════════════════════════════════════════════════════════════
elif page == "🎬 Video Detection":
    st.title("🎬 Video Detection")
    st.caption("Upload a video to detect license plates frame by frame")
    st.divider()

    detector = load_detector()
    if detector is None:
        st.stop()

    uploaded_video = st.file_uploader(
        "Upload Video",
        type = ["mp4", "avi", "mov", "mkv"],
        help = "Supported: MP4, AVI, MOV, MKV"
    )

    if uploaded_video is not None:
        # Video settings
        col1, col2 = st.columns(2)
        with col1:
            skip_frames = st.slider(
                "Process every N frames",
                min_value = 1,
                max_value = 30,
                value     = 5,
                help      = "Higher = faster but less accurate"
            )
        with col2:
            max_frames = st.slider(
                "Max frames to process",
                min_value = 50,
                max_value = 500,
                value     = 200,
                help      = "Limit frames to avoid long wait"
            )

        if st.button("🎬 Start Video Detection", type="primary", use_container_width=True):

            # Save uploaded video to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name

            # Open video
            cap          = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            all_results  = []
            frame_count  = 0

            # UI elements
            st.subheader("Processing Video...")
            progress_bar   = st.progress(0)
            status_text    = st.empty()
            frame_display  = st.empty()
            results_display = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break

                frame_count += 1
                progress = min(frame_count / min(total_frames, max_frames), 1.0)
                progress_bar.progress(progress)
                status_text.text(
                    f"Processing frame {frame_count}/"
                    f"{min(total_frames, max_frames)}..."
                )

                # Only run detection every N frames
                if frame_count % skip_frames == 0:
                    annotated, results = run_detection_on_frame(
                        frame, detector, source="video"
                    )

                    # Show current frame
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_display.image(
                        annotated_rgb,
                        caption      = f"Frame {frame_count}",
                        use_column_width = True
                    )

                    # Collect new results
                    for r in results:
                        if not r["is_duplicate"]:
                            all_results.append(r)
                            results_display.success(
                                f"🚗 New plate found: "
                                f"**{r['plate_text']}** "
                                f"(conf: {r['confidence']:.2f})"
                            )

            cap.release()
            os.unlink(tmp_path)  # delete temp file

            progress_bar.progress(1.0)
            status_text.text("✅ Video processing complete!")

            st.divider()
            st.subheader(f"Results — {len(all_results)} unique plates found")
            show_detection_results(all_results)


# ════════════════════════════════════════════════════════════
# PAGE 4 — WEBCAM DETECTION
# ════════════════════════════════════════════════════════════
elif page == "📷 Webcam Detection":
    st.title("📷 Webcam Detection")
    st.caption("Use your PC camera to detect license plates live")
    st.divider()

    detector = load_detector()
    if detector is None:
        st.stop()

    # Camera settings
    col1, col2 = st.columns(2)
    with col1:
        camera_index = st.selectbox(
            "Select Camera",
            options = [0, 1, 2],
            index   = 0,
            help    = "0 = built-in camera, 1 = external camera"
        )
    with col2:
        num_captures = st.slider(
            "Number of captures",
            min_value = 1,
            max_value = 20,
            value     = 5,
            help      = "How many frames to capture and analyze"
        )

    st.info("""
    **How to use:**
    1. Click **Start Webcam Detection** below
    2. Hold your number plate in front of the camera
    3. The system will automatically capture and detect plates
    4. Results appear below after all captures complete
    """)

    if st.button("📷 Start Webcam Detection", type="primary", use_container_width=True):

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(
                f"Cannot open camera {camera_index}. "
                "Make sure your webcam is connected and try camera index 1."
            )
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            all_results    = []
            frame_display  = st.empty()
            status_text    = st.empty()
            progress_bar   = st.progress(0)

            st.subheader("📸 Capturing frames...")

            for i in range(num_captures):
                # Warm up camera — read a few frames first
                for _ in range(5):
                    cap.read()

                ret, frame = cap.read()

                if not ret:
                    st.warning(f"Failed to capture frame {i+1}")
                    continue

                status_text.text(
                    f"Capturing {i+1}/{num_captures}..."
                )
                progress_bar.progress((i + 1) / num_captures)

                # Run detection
                annotated, results = run_detection_on_frame(
                    frame, detector, source="webcam"
                )

                # Show captured frame
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_display.image(
                    annotated_rgb,
                    caption          = f"Capture {i+1}/{num_captures}",
                    use_column_width = True,
                )

                for r in results:
                    if not r["is_duplicate"]:
                        all_results.append(r)

                # Small delay between captures
                time.sleep(0.5)

            cap.release()
            status_text.text("✅ Webcam capture complete!")

            st.divider()
            st.subheader(f"Results — {len(all_results)} unique plates found")
            show_detection_results(all_results)

            # Save screenshot of last frame
            if all_results:
                save_screenshot(
                    annotated,
                    all_results[0]["plate_text"],
                    "webcam"
                )


# ════════════════════════════════════════════════════════════
# PAGE 5 — DETECTION HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📋 Detection History":
    st.title("📋 Detection History")
    st.divider()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_dup = st.checkbox("Show duplicates", value=False)
    with col2:
        source_filter = st.multiselect(
            "Filter by source",
            options  = ["webcam", "image", "video"],
            default  = ["webcam", "image", "video"],
        )
    with col3:
        search = st.text_input("🔍 Search plate", placeholder="e.g. LHR-1234")

    # Load data
    df = get_all_detections(limit=1000)

    if not df.empty:
        if not show_dup:
            df = df[df["is_duplicate"] == 0]
        if source_filter:
            df = df[df["source"].isin(source_filter)]
        if search:
            df = df[df["plate_text"].str.contains(
                search.upper(), na=False
            )]

    if st.button("🔄 Refresh"):
        st.rerun()

    if not df.empty:
        st.dataframe(
            df[[
                "id", "plate_text", "confidence",
                "timestamp", "source",
                "vehicle_conf", "is_duplicate"
            ]],
            use_container_width = True,
            height              = 500,
            hide_index          = True,
            column_config       = {
                "plate_text"  : st.column_config.TextColumn("Plate", width=120),
                "confidence"  : st.column_config.ProgressColumn("OCR Conf",  max_value=1),
                "vehicle_conf": st.column_config.ProgressColumn("YOLO Conf", max_value=1),
                "is_duplicate": st.column_config.CheckboxColumn("Duplicate"),
            }
        )

        # Download
        csv_data = df.to_csv(index=False)
        st.download_button(
            label     = "⬇️ Download CSV",
            data      = csv_data,
            file_name = f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime      = "text/csv",
        )
    else:
        st.info("No records found for selected filters.")

    st.divider()

    # Unique plates table
    st.subheader("🚗 Unique Plates Summary")
    unique_df = get_unique_plates()
    if not unique_df.empty:
        st.dataframe(
            unique_df,
            use_container_width = True,
            hide_index          = True,
        )


# ════════════════════════════════════════════════════════════
# PAGE 6 — ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Analytics")
    st.divider()

    df = get_all_detections(limit=5000)

    if df.empty:
        st.info("No data yet. Run some detections first.")
        st.stop()

    # Row 1 — Pie + Line
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detections by Source")
        source_counts = df["source"].value_counts().reset_index()
        source_counts.columns = ["source", "count"]
        fig = px.pie(
            source_counts,
            values = "count",
            names  = "source",
            hole   = 0.4,
            color_discrete_sequence = px.colors.qualitative.Set2,
        )
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Detections Over Time")
        df_time            = df.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"])
        df_time["hour"]    = df_time["timestamp"].dt.floor("H")
        hourly             = df_time.groupby("hour").size().reset_index(name="count")
        fig2 = px.line(
            hourly, x="hour", y="count",
            markers=True,
            color_discrete_sequence=["#1f77b4"],
        )
        fig2.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2 — Confidence histogram
    st.subheader("OCR Confidence Distribution")
    fig3 = px.histogram(
        df, x="confidence",
        nbins    = 20,
        range_x  = [0, 1],
        color_discrete_sequence = ["#2ecc71"],
    )
    fig3.update_layout(height=250, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Row 3 — Top plates bar chart
    st.subheader("Most Seen Plates")
    top_plates = (
        df.groupby("plate_text")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    fig4 = px.bar(
        top_plates,
        x = "plate_text",
        y = "count",
        color_discrete_sequence = ["#e74c3c"],
    )
    fig4.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig4, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "🇵🇰 Pakistani Smart ANPR System | "
    "YOLOv8 + EasyOCR + Streamlit | "
    f"Model: {'✅ Ready' if Path('models/best.pt').exists() else '❌ Not trained'}"
)