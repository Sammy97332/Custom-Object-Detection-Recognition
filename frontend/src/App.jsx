import { useEffect, useRef, useState } from "react";

const API_URL = "http://127.0.0.1:5000/predict";

export default function App() {
  const [mode, setMode] = useState("upload"); // "upload" | "webcam"
  const [conf, setConf] = useState(0.5);
  const [detections, setDetections] = useState([]);
  const [inferenceMs, setInferenceMs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [webcamActive, setWebcamActive] = useState(false);
  const [error, setError] = useState("");

  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const drawBoxes = (boxes, origW, origH, displayW, displayH) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = displayW;
    canvas.height = displayH;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    const scaleX = displayW / origW;
    const scaleY = displayH / origH;

    boxes.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const sx = x1 * scaleX;
      const sy = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;

      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3;
      ctx.strokeRect(sx, sy, sw, sh);

      const text = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = "bold 14px sans-serif";
      const textWidth = ctx.measureText(text).width;
      const badgeY = sy > 26 ? sy - 24 : sy;

      ctx.fillStyle = "#10b981";
      ctx.fillRect(sx, badgeY, textWidth + 12, 24);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, sx + 6, badgeY + 17);
    });
  };

  const runInference = async (blob, displayW, displayH) => {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    formData.append("conf", conf);

    try {
      setError("");
      const res = await fetch(API_URL, { method: "POST", body: formData });
      const data = await res.json();
      if (data.detections) {
        setDetections(data.detections);
        setInferenceMs(data.inference_ms);
        drawBoxes(
          data.detections,
          data.image_width,
          data.image_height,
          displayW,
          displayH
        );
      }
    } catch {
      setError("Cannot reach Flask backend at http://127.0.0.1:5000. Make sure backend/app.py is running.");
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setDetections([]);
    setLoading(true);
    const url = URL.createObjectURL(file);
    const img = imgRef.current;
    img.src = url;

    img.onload = async () => {
      await runInference(file, img.clientWidth, img.clientHeight);
      setLoading(false);
    };
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setWebcamActive(true);

      const captureCanvas = document.createElement("canvas");
      intervalRef.current = setInterval(() => {
        const video = videoRef.current;
        if (!video || video.readyState !== 4) return;

        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCanvas.getContext("2d").drawImage(video, 0, 0);

        captureCanvas.toBlob(
          (blob) => {
            if (blob) {
              runInference(blob, video.clientWidth, video.clientHeight);
            }
          },
          "image/jpeg",
          0.8
        );
      }, 300);
    } catch {
      setError("Unable to access webcam. Check browser camera permissions.");
    }
  };

  const stopWebcam = () => {
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setWebcamActive(false);
    setDetections([]);
  };

  useEffect(() => () => stopWebcam(), []);

  return (
    <div style={{ maxWidth: 820, margin: "2rem auto", fontFamily: "sans-serif", padding: "0 1rem" }}>
      <h1>Custom Facial Recognition System</h1>

      <div style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap" }}>
        <button
          onClick={() => { stopWebcam(); setMode("upload"); }}
          style={{ padding: "0.5rem 1rem", fontWeight: mode === "upload" ? "bold" : "normal", cursor: "pointer" }}
        >
          Upload Image
        </button>
        <button
          onClick={() => setMode("webcam")}
          style={{ padding: "0.5rem 1rem", fontWeight: mode === "webcam" ? "bold" : "normal", cursor: "pointer" }}
        >
          Live Webcam
        </button>

        <label style={{ marginLeft: "auto" }}>
          Confidence: <strong>{(conf * 100).toFixed(0)}%</strong>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>
      </div>

      {mode === "upload" ? (
        <input type="file" accept="image/*" onChange={handleFileUpload} />
      ) : (
        <button onClick={webcamActive ? stopWebcam : startWebcam} style={{ padding: "0.5rem 1rem", cursor: "pointer" }}>
          {webcamActive ? "Stop Camera" : "Start Camera"}
        </button>
      )}

      {error && (
        <p style={{ color: "#dc2626", marginTop: "0.75rem", fontWeight: "bold" }}>{error}</p>
      )}

      <div style={{ position: "relative", marginTop: "1rem", display: "inline-block" }}>
        {mode === "upload" ? (
          <img ref={imgRef} alt="" style={{ maxWidth: "100%", display: "block", borderRadius: 8 }} />
        ) : (
          <video ref={videoRef} autoPlay playsInline muted style={{ maxWidth: "100%", display: "block", borderRadius: 8 }} />
        )}
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
        />
      </div>

      <div style={{ marginTop: "1rem", padding: "0.75rem", background: "#f3f4f6", borderRadius: 8 }}>
        {loading ? (
          <span>Running inference...</span>
        ) : (
          <span>
            Status:{" "}
            <strong>
              {detections.length > 0
                ? `Target Identified (${detections.length} match)`
                : "No Target Detected"}
            </strong>
            {inferenceMs !== null && ` • Inference time: ${inferenceMs} ms`}
          </span>
        )}
      </div>
    </div>
  );
}