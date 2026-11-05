import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import yaml
from ultralytics import YOLO


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Fridge Bottle Counting",
    page_icon="🧊",
    layout="wide",
)

# ------------------------------------------------------------
# DARK DASHBOARD DESIGN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background: #101827; color: #F1F5F9; }
    [data-testid="stSidebar"] { background: #172235; }
    [data-testid="stMetric"] {
        background: #1D2B40;
        border: 1px solid #30435C;
        padding: 16px;
        border-radius: 12px;
    }
    h1, h2, h3 { color: #F8FAFC; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# LOAD YOLO MODEL ONCE
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    # YOLOv8n pretrained model; COCO class 39 is bottle.
    return YOLO("yolov8n.pt")


# ------------------------------------------------------------
# CREATE BYTE TRACKER CONFIGURATION
# ------------------------------------------------------------
def create_tracker_config(folder):
    tracker_config = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.35,
        "track_buffer": 60,
        "match_thresh": 0.80,
        "fuse_score": True,
    }
    tracker_path = os.path.join(folder, "bytetrack.yaml")
    with open(tracker_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(tracker_config, file)
    return tracker_path


# ------------------------------------------------------------
# VIDEO PROCESSING AND AUTOMATIC INVENTORY COUNTING
# ------------------------------------------------------------
def process_video(
    input_path,
    output_path,
    initial_inventory,
    roi_x1,
    roi_y1,
    roi_x2,
    roi_y2,
    confidence,
    confirm_frames,
):
    model = load_model()
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Keep the selected ROI inside the actual video dimensions.
    rx1 = max(0, min(int(roi_x1), width - 1))
    ry1 = max(0, min(int(roi_y1), height - 1))
    rx2 = max(0, min(int(roi_x2), width - 1))
    ry2 = max(0, min(int(roi_y2), height - 1))

    if rx1 >= rx2 or ry1 >= ry2:
        cap.release()
        raise ValueError(
            f"Invalid ROI. Video size is {width}x{height}; "
            "right/bottom coordinates must be greater than left/top."
        )

    inventory = int(initial_inventory)
    entries = 0
    exits = 0
    frame_number = 0

    # A new tracking ID is only a baseline. It is not counted immediately.
    track_data = {}
    event_log = []

    temp_dir = os.path.dirname(output_path)
    tracker_path = create_tracker_config(temp_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the output video.")

    progress = st.progress(0)
    status = st.empty()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            current_ids = set()

            # Run YOLOv8n bottle detection and ByteTrack tracking.
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_path,
                conf=confidence,
                iou=0.65,
                classes=[39],
                imgsz=640,
                device="cpu",
                verbose=False,
            )

            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = (
                        box.xyxy[0].cpu().numpy().astype(int)
                    )
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    inside_now = (
                        rx1 <= center_x <= rx2
                        and ry1 <= center_y <= ry2
                    )
                    current_ids.add(int(track_id))

                    # First observation establishes the baseline state.
                    if int(track_id) not in track_data:
                        track_data[int(track_id)] = {
                            "state": inside_now,
                            "pending_state": inside_now,
                            "pending_count": 0,
                        }
                    else:
                        info = track_data[int(track_id)]
                        previous_state = info["state"]

                        if inside_now != previous_state:
                            if info["pending_state"] == inside_now:
                                info["pending_count"] += 1
                            else:
                                info["pending_state"] = inside_now
                                info["pending_count"] = 1

                            # Confirm a crossing only after consecutive frames.
                            if info["pending_count"] >= confirm_frames:
                                if previous_state and not inside_now:
                                    inventory = max(0, inventory - 1)
                                    exits += 1
                                    event_type = "EXIT"
                                    change = "-1"
                                else:
                                    inventory += 1
                                    entries += 1
                                    event_type = "ENTER"
                                    change = "+1"

                                event_log.append({
                                    "Frame": frame_number,
                                    "Tracking ID": int(track_id),
                                    "Event": event_type,
                                    "Change": change,
                                    "Inventory": inventory,
                                })

                                info["state"] = inside_now
                                info["pending_count"] = 0
                                info["pending_state"] = inside_now
                        else:
                            info["pending_count"] = 0
                            info["pending_state"] = inside_now

                    color = (0, 255, 0) if inside_now else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id} | "
                        f"{'INSIDE' if inside_now else 'OUTSIDE'}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            # Missing detections do not mean that a bottle exited.
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                "FRIDGE COUNTING AREA",
                (rx1, max(20, ry1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Fridge Inventory: {inventory}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Entries: {entries} | Exits: {exits}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Visible bottles: {len(current_ids)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            writer.write(frame)
            frame_number += 1

            if total_frames > 0:
                progress.progress(min(frame_number / total_frames, 1.0))
            status.write(
                f"Processed {frame_number}/{total_frames or '?'} frames "
                f"| Inventory: {inventory}"
            )

    finally:
        cap.release()
        writer.release()

    progress.progress(1.0)
    status.success(f"Video processing completed: {frame_number} frames.")

    events_df = pd.DataFrame(
        event_log,
        columns=["Frame", "Tracking ID", "Event", "Change", "Inventory"],
    )

    return {
        "initial": int(initial_inventory),
        "entries": entries,
        "exits": exits,
        "final": inventory,
        "frames": frame_number,
        "events": events_df,
        "video_path": output_path,
        "video_width": width,
        "video_height": height,
    }


# ------------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------------
st.sidebar.title("🧊 Smart Fridge AI")
st.sidebar.caption("Bottle Inventory Management")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Video Monitoring", "About Project"],
)

st.sidebar.divider()
st.sidebar.subheader("Video Processing Settings")

initial_inventory = st.sidebar.number_input(
    "Starting bottles in fridge",
    min_value=0,
    value=5,
    step=1,
)
confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
)
confirm_frames = st.sidebar.number_input(
    "Frames to confirm entry/exit",
    min_value=1,
    max_value=15,
    value=3,
    step=1,
)

st.sidebar.caption(
    "The default ROI coordinates match the coordinates in your Colab code. "
    "Adjust them for each uploaded video."
)


# ------------------------------------------------------------
# MAIN DASHBOARD
# ------------------------------------------------------------
st.title("Smart Refrigerator Bottle Counting")
st.caption(
    "YOLOv8n + ByteTrack | Automatic bottle entry and exit monitoring"
)

if "result" not in st.session_state:
    st.session_state.result = None

if page == "Dashboard":
    result = st.session_state.result

    if result is None:
        st.info(
            "Upload a fridge video in Video Monitoring and process it "
            "to display your inventory results here."
        )
        st.metric("Starting inventory", initial_inventory)
        st.metric("Current inventory", initial_inventory)
    else:
        st.success("Latest video analysis completed.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Starting Inventory", result["initial"])
        c2.metric("Bottle Entries", result["entries"], delta=f'+{result["entries"]}')
        c3.metric("Bottle Exits", result["exits"], delta=f'-{result["exits"]}')
        c4.metric("Current Inventory", result["final"])

        st.subheader("Bottle Movement Analysis")
        chart_df = pd.DataFrame({
            "Movement": ["Entries", "Exits"],
            "Bottles": [result["entries"], result["exits"]],
        })
        st.bar_chart(chart_df, x="Movement", y="Bottles")

        st.subheader("Recent Inventory Events")
        if result["events"].empty:
            st.info("No confirmed entry or exit events were detected.")
        else:
            st.dataframe(
                result["events"],
                use_container_width=True,
                hide_index=True,
            )
            csv_data = result["events"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download event history (CSV)",
                data=csv_data,
                file_name="bottle_inventory_events.csv",
                mime="text/csv",
            )


elif page == "Video Monitoring":
    st.subheader("Upload and Process Fridge Video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    st.markdown("#### Fridge Region of Interest (ROI)")
    roi_col1, roi_col2 = st.columns(2)
    with roi_col1:
        roi_x1 = st.number_input("Left (X1)", min_value=0, value=165, step=1)
        roi_y1 = st.number_input("Top (Y1)", min_value=0, value=400, step=1)
    with roi_col2:
        roi_x2 = st.number_input("Right (X2)", min_value=1, value=300, step=1)
        roi_y2 = st.number_input("Bottom (Y2)", min_value=1, value=650, step=1)

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Process video and count bottles", type="primary"):
            with st.spinner("Loading YOLOv8n and processing the video..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        input_path = os.path.join(temp_dir, uploaded_video.name)
                        output_path = os.path.join(temp_dir, "bottle_inventory_result.mp4")

                        with open(input_path, "wb") as video_file:
                            video_file.write(uploaded_video.getbuffer())

                        result = process_video(
                            input_path=input_path,
                            output_path=output_path,
                            initial_inventory=initial_inventory,
                            roi_x1=roi_x1,
                            roi_y1=roi_y1,
                            roi_x2=roi_x2,
                            roi_y2=roi_y2,
                            confidence=confidence,
                            confirm_frames=int(confirm_frames),
                        )

                        # Keep the processed video bytes after the temporary folder closes.
                        with open(output_path, "rb") as output_file:
                            processed_video_bytes = output_file.read()

                        result["video_bytes"] = processed_video_bytes
                        st.session_state.result = result

                    st.success("Bottle counting completed.")
                    st.video(st.session_state.result["video_bytes"])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Entries", result["entries"])
                    c2.metric("Exits", result["exits"])
                    c3.metric("Final inventory", result["final"])

                    if result["events"].empty:
                        st.info("No confirmed movement events were detected.")
                    else:
                        st.dataframe(
                            result["events"],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download event history (CSV)",
                            data=result["events"].to_csv(index=False).encode("utf-8"),
                            file_name="bottle_inventory_events.csv",
                            mime="text/csv",
                        )

                except Exception as error:
                    st.error(f"Video processing failed: {error}")
    else:
        st.info("Upload a fridge video to begin.")


elif page == "About Project":
    st.subheader("About this project")
    st.write(
        "This application analyzes an uploaded refrigerator video, "
        "detects bottles with YOLOv8n, tracks them with ByteTrack, "
        "and updates inventory when a tracked bottle crosses the "
        "configured refrigerator ROI boundary."
    )
    st.markdown(
        """
        **Technology stack**
        - YOLOv8n: bottle detection (COCO class 39)
        - ByteTrack: object tracking
        - OpenCV: video processing
        - Streamlit: web dashboard
        - Pandas: inventory event history
        """
    )
    st.warning(
        "The app uses the same ROI-crossing approach as your Colab Cell 30. "
        "A new tracking ID is treated as a baseline, not an entry. "
        "The ROI must be adjusted to match the uploaded video's fridge area."
    )

st.divider()
st.caption("Smart Refrigerator Bottle Counting | Computer Vision Project")
