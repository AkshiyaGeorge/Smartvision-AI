import streamlit as st
import streamlit as st
from PIL import Image
import os

st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for fonts, colors, buttons
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
    }
    h2 {
        color: #34495e;
        font-size: 28px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.write("Welcome to SmartVision AI 🤖. Use the navigation menu above to explore features.")


# -----------------------------
# Page 1: Home
# -----------------------------
import streamlit as st
import os
import cv2
import numpy as np

CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "traffic light","stop sign","bench","bird","cat","dog","horse","cow",
    "elephant","bottle","cup","bowl","pizza","cake","chair","couch","potted plant","bed"
]

def draw_yolo_boxes(image_path, label_path, class_names):
    """Draw precise YOLO bounding boxes with small, clear labels."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x_center, y_center, bw, bh = map(float, parts)
            cls_id = int(cls_id)

            label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

            # Convert normalized YOLO coords to pixel coords
            bw_px = int(bw * w)
            bh_px = int(bh * h)
            x_px = int(x_center * w)
            y_px = int(y_center * h)

            x1 = max(x_px - bw_px // 2, 0)
            y1 = max(y_px - bh_px // 2, 0)
            x2 = min(x_px + bw_px // 2, w - 1)
            y2 = min(y_px + bh_px // 2, h - 1)

            # Draw tight rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            # Small, crisp label
            font_scale = 0.5
            thickness = 1
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_w, label_h = label_size
            cv2.rectangle(img, (x1, y1 - label_h - 4), (x1 + label_w + 6, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 3, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def page_home():
    st.markdown("<h1 style='color:#2c3e50;'>🤖 SmartVision AI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#2980b9;'>✨ Welcome to SmartVision AI</h3>", unsafe_allow_html=True)
    st.write("Use the navigation menu above to explore features.")

    st.markdown("<h2 style='color:#16a085;'>📌 Project Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#34495e; font-size:16px;'>
    SmartVision AI is a unified platform for image classification and object detection.<br><br>
    <b>🔍 Key Features:</b><br>
    • 🧠 Image classification with 4 CNN models: <span style='color:#8e44ad;'>VGG16</span>, <span style='color:#8e44ad;'>ResNet50</span>, <span style='color:#8e44ad;'>MobileNetV2</span>, <span style='color:#8e44ad;'>EfficientNetB0</span><br>
    • 🎯 Object detection with <span style='color:#e74c3c;'>YOLOv8</span><br>
    • 📊 Model performance dashboards<br>
    • 📷 Optional live webcam detection
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#d35400;'>🖼️ Sample Detection Demos</h2>", unsafe_allow_html=True)

    # Replace last two unclear demos with new ones
    demo_images = [
        ("smartvision_dataset/detection/images/image_000000.jpg",
         "smartvision_dataset/detection/labels/image_000000.txt",
         "📷 Demo 1: Street Scene"),
        ("smartvision_dataset/detection/images/image_000010.jpg",
         "smartvision_dataset/detection/labels/image_000010.txt",
         "📷 Demo 2: Restaurant Scene"),
        ("smartvision_dataset/detection/images/image_000020.jpg",
         "smartvision_dataset/detection/labels/image_000020.txt",
         "📷 Demo 3: Outdoor Bench Scene")
    ]

    for img_path, label_path, caption in demo_images:
        if os.path.exists(img_path):
            annotated_img = draw_yolo_boxes(img_path, label_path, CLASS_NAMES)
            if annotated_img is not None:
                st.image(annotated_img, caption=caption, use_container_width=True)
            else:
                st.image(img_path, caption=f"{caption} (no annotations)", use_container_width=True)
        else:
            st.warning(f"⚠️ Demo image not found: {img_path}")


# -----------------------------
# Page 2: Image Classification — Upload, predict with 4 CNNs, show Top-5 + comparison
# -----------------------------
import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import gdown

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

# -----------------------------
# Config: Google Drive IDs and class labels
# -----------------------------
# Replace with your actual Drive file IDs
DRIVE_IDS = {
    "VGG16": "1Pn7IDM2pJdg8E5y4kQjzdJ-1criM4UvB",
    "ResNet50": "1X-i_G_9Kyl7rNGbmEDQ5yYHF_SyzRwVW",
    "MobileNetV2": "1_TxFmwxdeiaVXfAAYWWfnEKUUSd_gv_K",
    "EfficientNetB0": "1vK88TFwcavnPi2n-vms5WE8DhLTDcwBc",
}

CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'traffic light','stop sign','bench','bird','cat','dog','horse','cow',
    'elephant','bottle','cup','bowl','pizza','cake','chair','couch','potted plant','bed'
]

# -----------------------------
# Helper: download model from Google Drive if not present
# -----------------------------
base_path = "Smartvision-AI/models"
os.makedirs(base_path, exist_ok=True)

def ensure_model(file_id, filename):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return path

# -----------------------------
# Cached model loading
# -----------------------------
@st.cache_resource
def get_model(path: str):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    else:
        st.error(f"❌ Model file not found: {path}")
        return None

# -----------------------------
# Preprocessing and inference
# -----------------------------
def predict_top5(model, img: Image.Image, class_names: list, pre_fn):
    input_shape = model.input_shape
    img_size = (input_shape[1], input_shape[2])

    arr = np.array(img.resize(img_size)).astype(np.float32)
    arr = pre_fn(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idxs = np.argsort(probs)[-5:][::-1]
    return [(class_names[i], float(probs[i])) for i in idxs]

# -----------------------------
# Streamlit page
# -----------------------------
def page_classification():
    st.markdown("<h1 style='color:#8e44ad;'>🧠 Image Classification — Real-time</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="classification_uploader")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.subheader("Predictions from CNN Models (Top-5)")
        top1_labels, top1_confs = [], []

        # Show predictions from all 4 models
        for name, file_id in DRIVE_IDS.items():
            # Ensure model is downloaded
            model_path = ensure_model(file_id, f"best_{name.lower()}.keras")
            model = get_model(model_path)

            if model:
                preds = predict_top5(model, img, CLASS_NAMES, {
                    "VGG16": vgg_pre,
                    "ResNet50": res_pre,
                    "MobileNetV2": mob_pre,
                    "EfficientNetB0": eff_pre
                }[name])

                st.write(f"**{name}**")
                df = pd.DataFrame(preds, columns=["Label", "Confidence"])
                df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").round(4)
                st.table(df)

                # Save Top-1 for comparison
                top1_labels.append(df.iloc[0]["Label"])
                top1_confs.append(df.iloc[0]["Confidence"])
            else:
                top1_labels.append(None)
                top1_confs.append(None)

        # Side-by-side model comparison
        st.subheader("Model Comparison — Top-1 Prediction")
        comparison = {
            "Model": list(DRIVE_IDS.keys()),
            "Top-1 Label": top1_labels,
            "Top-1 Confidence": top1_confs
        }
        st.table(pd.DataFrame(comparison))

# -----------------------------
# Page 3: Object Detection — Real-time YOLOv8
# -----------------------------
import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model once (cached)
@st.cache_resource
def load_yolo_model():
    weights_path = "yolov8s.pt"  # adjust if using custom-trained weights
    if os.path.exists(weights_path):
        return YOLO(weights_path)
    else:
        st.error(f"⚠️ YOLO weights not found at {weights_path}")
        return None

def page_detection():
    st.markdown(
        "<h1 style='color:#e74c3c;'>🎯 Object Detection (YOLOv8)</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='color:#2980b9;'>Upload an image to detect objects in real-time</h3>",
        unsafe_allow_html=True
    )

    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="detection_conf_slider")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="detection_uploader")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.subheader("Detections")

        # Load YOLO model
        yolo_model = load_yolo_model()
        if yolo_model is not None:
            results = yolo_model.predict(source=img, conf=conf_thresh, verbose=False)

            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = yolo_model.names[cls_id]
                        detections.append({"Object": label, "Confidence": round(conf, 4)})

            # Show detection table
            if detections:
                st.table(detections)

                # ✅ Show image with bounding boxes + labels
                # results[0].plot() returns a numpy array with boxes drawn
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="Detections with Bounding Boxes", use_container_width=True)
            else:
                st.info("No objects detected at this confidence threshold.")


# -----------------------------
# Page 4: Model Performance
# -----------------------------
import os
import streamlit as st
import pandas as pd

# Best epoch results for each model
results = [
    {"Model": "VGG16", "Epoch": 20, "acc": 65.7, "loss": 1.168, "precision": 83.6, "recall": 51.9,
     "top5": 90.3, "val_acc": 37.2, "val_loss": 2.211, "val_precision": 66.8, "val_recall": 23.6,
     "val_top5": 72.6, "lr": 0.00025},
    {"Model": "ResNet50", "Epoch": 20, "acc": 100.0, "loss": 0.002, "precision": 100.0, "recall": 100.0,
     "top5": 100.0, "val_acc": 41.9, "val_loss": 2.742, "val_precision": 50.5, "val_recall": 37.4,
     "val_top5": 72.9, "lr": 0.00003125},
    {"Model": "MobileNetV2", "Epoch": 12, "acc": 94.5, "loss": 0.222, "precision": 97.6, "recall": 91.0,
     "top5": 99.2, "val_acc": 36.2, "val_loss": 2.830, "val_precision": 50.3, "val_recall": 29.2,
     "val_top5": 63.9, "lr": 0.00025},
    {"Model": "EfficientNetB0", "Epoch": 14, "acc": 97.5, "loss": 0.122, "precision": 98.7, "recall": 95.6,
     "top5": 99.6, "val_acc": 41.9, "val_loss": 2.603, "val_precision": 50.6, "val_recall": 37.1,
     "val_top5": 70.9, "lr": 0.000075}
]

def page_performance():
    # Styled title with emoji
    st.markdown(
        "<h1 style='color:#27ae60;'>📊 Model Performance Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='color:#2980b9;'>Compare accuracy, precision, recall, F1, inference speed, and confusion matrices</h3>",
        unsafe_allow_html=True
    )

    df_results = pd.DataFrame(results)

    # ✅ Accuracy comparison
    st.markdown("<h2 style='color:#8e44ad;'>✅ Accuracy Comparison</h2>", unsafe_allow_html=True)
    st.bar_chart(df_results.set_index("Model")[["acc", "val_acc"]])

    # ✅ Precision/Recall comparison
    st.markdown("<h2 style='color:#2c3e50;'>🎯 Precision & Recall</h2>", unsafe_allow_html=True)
    st.bar_chart(df_results.set_index("Model")[["precision", "recall", "val_precision", "val_recall"]])

    # ✅ Top-5 accuracy comparison
    st.markdown("<h2 style='color:#9b59b6;'>🏆 Top-5 Accuracy</h2>", unsafe_allow_html=True)
    st.bar_chart(df_results.set_index("Model")[["top5", "val_top5"]])

    # ✅ Inference speed comparison (hard-coded example, replace with measured times)
    st.markdown("<h2 style='color:#d35400;'>⚡ Inference Speed (seconds per image)</h2>", unsafe_allow_html=True)
    st.bar_chart({"VGG16": 0.12, "ResNet50": 0.18, "MobileNetV2": 0.08, "EfficientNetB0": 0.10})

    # 📑 Class-wise performance breakdown
    st.markdown("<h2 style='color:#16a085;'>📑 Class-wise Performance</h2>", unsafe_allow_html=True)
    if os.path.exists("class_metrics.csv"):
        df_classes = pd.read_csv("class_metrics.csv")
        st.dataframe(df_classes.style.highlight_max(axis=0, color="lightgreen"))

        # Per-class comparison chart
        selected_class = st.selectbox("Select a class to compare across models:", df_classes["Class"])
        row = df_classes[df_classes["Class"] == selected_class].iloc[0]

        chart_data = pd.DataFrame({
            "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
            "Precision": [row["VGG16_Precision"], row["ResNet50_Precision"], row["MobileNetV2_Precision"], row["EffNetB0_Precision"]],
            "Recall": [row["VGG16_Recall"], row["ResNet50_Recall"], row["MobileNetV2_Recall"], row["EffNetB0_Recall"]],
            "F1": [row["VGG16_F1"], row["ResNet50_F1"], row["MobileNetV2_F1"], row["EffNetB0_F1"]],
        })
        st.bar_chart(chart_data.set_index("Model"))
    else:
        st.warning("⚠️ class_metrics.csv not found. Please export per-class metrics.")

    # 🌀 Confusion matrices
    st.markdown("<h2 style='color:#c0392b;'>🌀 Confusion Matrices</h2>", unsafe_allow_html=True)
    cm_files = {
        "VGG16": "confusion matrix VGG16.png",
        "ResNet50": "confusion matrix ResNet50.png",
        "MobileNetV2": "confusion matrix MobileNetV2.png",
        "EfficientNetB0": "confusion matrix EfficientNetB0.png"
    }

    for model_name, file_name in cm_files.items():
        if os.path.exists(file_name):
            st.image(file_name, caption=f"📈 Confusion Matrix — {model_name}", use_container_width=True)
        else:
            st.warning(f"⚠️ Confusion matrix image not found for {model_name}: {file_name}")


# -----------------------------
# Page 5: Live Webcam Detection (YOLOv8)
# -----------------------------
import cv2
import time
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Cache YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8s.pt")  # replace with your trained weights if needed

def page_webcam():
    st.markdown(
        "<h1 style='color:#d35400;'>📹 Live Webcam Detection (YOLOv8)</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='color:#2980b9;'>Real-time detection with FPS and latency metrics</h3>",
        unsafe_allow_html=True
    )

    conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, key="webcam_conf_slider")


    # Start webcam button
    if st.button("Start Webcam Detection", key="webcam_start_button"):

        yolo_model = load_yolo_model()
        cap = cv2.VideoCapture(0)  # 0 = default webcam

        stframe = st.empty()  # placeholder for video frames
        fps_display = st.empty()
        latency_display = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Unable to access webcam.")
                break

            start = time.time()
            # Run YOLO inference
            results = yolo_model.predict(source=frame, conf=conf_thresh, verbose=False)

            # Annotate frame with detections
            annotated_frame = results[0].plot()  # YOLO provides .plot() for bounding boxes

            # Convert BGR (OpenCV) → RGB (Streamlit/PIL)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Show frame in Streamlit
            stframe.image(annotated_frame, caption="Live Detection", width=640)

            # FPS & latency
            elapsed = time.time() - start
            fps = 1 / elapsed if elapsed > 0 else 0
            fps_display.metric("FPS", f"{fps:.2f}")
            latency_display.metric("Latency (ms)", f"{elapsed*1000:.1f}")

            # Stop condition (Streamlit reruns script each interaction)
            if not cap.isOpened():
                break

        cap.release()
        cv2.destroyAllWindows()

# -----------------------------
# Page 6: About
# -----------------------------
import streamlit as st

def page_about():
    # Styled title
    st.markdown(
        "<h1 style='color:#2980b9;'>ℹ️ About SmartVision AI</h1>",
        unsafe_allow_html=True
    )

    # Project Documentation
    st.markdown("<h2 style='color:#16a085;'>📘 Project Documentation</h2>", unsafe_allow_html=True)
    st.write("""
    SmartVision AI is a unified platform for image classification and object detection.
    It integrates multiple CNN architectures for classification and YOLOv8 for detection,
    providing a complete workflow from dataset preparation to model benchmarking and deployment.
    """)

    # Dataset Information
    st.markdown("<h2 style='color:#8e44ad;'>📂 Dataset Information</h2>", unsafe_allow_html=True)
    st.write("""
    - **Source:** COCO dataset (25 selected classes)
    - **Classes Used:** person, bicycle, car, motorcycle, airplane, bus, train, truck,
      traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant,
      bottle, cup, bowl, pizza, cake, chair, couch, potted plant
    - **Images per class:** 100 (balanced dataset)
    - **Structure:** Train / Validation / Test splits with augmentation applied
    """)

    # Model Architectures
    st.markdown("<h2 style='color:#e74c3c;'>🧠 Model Architectures Used</h2>", unsafe_allow_html=True)
    st.write("""
    - **Classification Models:**
      - VGG16
      - ResNet50
      - MobileNetV2
      - EfficientNetB0
    - **Detection Model:**
      - YOLOv8 (Ultralytics implementation)
    """)

    # Technical Stack
    st.markdown("<h2 style='color:#d35400;'>⚙️ Technical Stack</h2>", unsafe_allow_html=True)
    st.write("""
    - **Languages:** Python
    - **Libraries:** TensorFlow/Keras, PyTorch, Ultralytics YOLO, OpenCV, Streamlit, scikit-learn, seaborn, matplotlib
    - **Tools:** Jupyter Notebook for experimentation, Streamlit for deployment
    - **Environment:** Windows 11, Python 3.13
    """)

    # Developer Information
    st.markdown("<h2 style='color:#27ae60;'>👩‍💻 Developer Information</h2>", unsafe_allow_html=True)
    st.write("""
    - **Developer:** Akshiya
    - **Role:** Data Scientist & ML Engineer @GUVI
    - **Focus:** Architecting data-driven financial risk and sports analytics platforms,
      building scalable ML pipelines, and deploying robust computer vision applications.
    - **Education:** MBA (HR), B.Tech (IT)
    """)

    st.success("SmartVision AI brings together classification, detection, and performance benchmarking into one cohesive application.")
# -----------------------------
# Main app with navigation
# -----------------------------
# Center Navigation with Tabs
# -----------------------------
import streamlit as st

PAGES = {
    "Home": page_home,
    "Image Classification": page_classification,
    "Object Detection": page_detection,
    "Model Performance": page_performance,
    "Live Webcam Detection": page_webcam,
    "About": page_about,
}

import streamlit as st

# Define your page functions
PAGES = {
    "🏠 Home": page_home,
    "🧠 Image Classification": page_classification,
    "🎯 Object Detection": page_detection,
    "📊 Model Performance": page_performance,
    "📹 Live Webcam Detection": page_webcam,
    "ℹ️ About": page_about,
}

def main():
    # Optional welcome banner
    st.markdown(
        "<h2 style='color:#2c3e50;'>Welcome to SmartVision AI</h2>",
        unsafe_allow_html=True
    )
    st.write("Use the tabs below to explore features.")

    # Create tabs with emojis
    tab_names = list(PAGES.keys())
    tabs = st.tabs(tab_names)

    # Render only the selected tab's content
    for i, name in enumerate(tab_names):
        with tabs[i]:
            PAGES[name]()  # Call the corresponding page function

if __name__ == "__main__":
    main()
