Custom Object Detection and Facial Recognition

📌 Project Overview
This project focuses on building a custom Object Detection and Recognition pipeline capable of detecting and identifying a specific target person (**Atsisco**) across varying environments, lighting conditions, and camera angles.

The project covers the end-to-end computer vision workflow: custom dataset collection, multi-folder data organization, bounding-box annotation, model training, validation, and real-time recognition testing.

📂 Dataset Structure & Storage
The raw image dataset is hosted on Google Drive across **7 image folders**, while bounding-box annotations and train/validation splits are managed on the **Ultralytics Platform**.

Google Drive Dataset Link:** [Insert Your Google Drive Folder Link Here]
Drive Organization:** 7 image batch folders (435 total images collected across varied lighting conditions, backgrounds, and camera angles)
Ultralytics Dataset Name:** `Atsisco` (License: `CC BY 4.0`)
Dataset Status:** `Ready` for training
Total Images:** 435 (377 labeled with 377 bounding-box annotations; 58 background images)
Target Class:** 1 class (`Atsisco`)
Data Split:** 399 Train (~91.7%) / 36 Validation (~8.3%)

```text
Object_Detection_Dataset/
├── Folder_1/                     # Image batch 1
├── Folder_2/                     # Image batch 2
├── Folder_3/                     # Image batch 3
├── Folder_4/                     # Image batch 4
├── Folder_5/                     # Image batch 5
├── Folder_6/                     # Image batch 6
└── Folder_7/                     # Image batch 7
```

---

🛠️ Tech Stack & Tools
Programming Language:** Python 3.12
IDE / Environments:** Visual Studio Code, Jupyter Notebook, Google Colab
Computer Vision & ML Libraries:** Ultralytics (`YOLO26`), OpenCV, NumPy
Annotation & Training Platform:** Ultralytics Platform

🚀 Project Roadmap
[x] **Phase 1:** Repository setup and Google Drive dataset architecture
[x] **Phase 2:** Personal images collected and organized across 7 Google Drive folders (435 total images)
[x] **Phase 3:** Image preprocessing, bounding-box annotation (377 labeled images), and Train/Val splitting (`399 Train` / `36 Val`)
[ ] **Phase 4:** Model training, validation (`mAP`), and real-time webcam/video recognition testing

⚡ Quickstart (Training & Inference)

1. Install Dependencies
```bash
pip install ultralytics opencv-python
```

2. Train the Model
```python
from ultralytics import YOLO

# Load a pretrained YOLO26 Nano model
model = YOLO("yolo26n.pt")

# Train on the Atsisco custom dataset
results = model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

3. Run Real-Time Recognition
```python
from ultralytics import YOLO

# Load best trained weights
model = YOLO("runs/detect/train/weights/best.pt")

# Run live webcam inference (source=0)
results = model.predict(source=0, conf=0.5, show=True)
```

---

## 👤 Author
* **Name:** Nkrumah Samuel Kojo
* **Course / Assignment:** Object Detection and Recognition
