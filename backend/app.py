from pathlib import Path
import time
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load your trained YOLO26n weights reliably from backend/models/best.pt
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best.pt"
model = YOLO(str(MODEL_PATH))


@app.route("/", methods=["GET"])
def index():
    return {
        "service": "VisionLab YOLO Inference API",
        "status": "online",
        "endpoints": ["/health", "/predict"]
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": model.names})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    conf_threshold = float(request.form.get("conf", 0.5))

    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    max_dim = 640
    h0, w0 = image.shape[:2]
    if max(h0, w0) > max_dim:
        scale = max_dim / float(max(h0, w0))
        image = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]

    start = time.perf_counter()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    inference_ms = round((time.perf_counter() - start) * 1000, 1)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "label": model.names[cls_id],
            "confidence": round(confidence, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return jsonify({
        "detections": detections,
        "count": len(detections),
        "image_width": width,
        "image_height": height,
        "inference_ms": inference_ms,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)