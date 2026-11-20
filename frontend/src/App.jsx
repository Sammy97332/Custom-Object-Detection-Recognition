import { useEffect, useRef, useState } from "react";
import "./App.css";

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
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24"><path d="M12 3l7 3v5c0 4.6-2.9 7.9-7 10-4.1-2.1-7-5.4-7-10V6l7-3z" /><circle cx="12" cy="11" r="3" /></svg>
          </div>
          <span>Vision<span>Lab</span></span>
        </div>

        <nav className="side-nav" aria-label="Primary navigation">
          <button className="nav-item active" type="button">
            <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></svg>
            Detection
          </button>
          <button className="nav-item" type="button">
            <svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1018 0 9 9 0 10-18 0"/><path d="M12 7v5l3 2"/></svg>
            History
          </button>
          <button className="nav-item" type="button">
            <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.6v.2H11V20a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1L6.2 17l.1-.1a1.7 1.7 0 00.3-1.9A1.7 1.7 0 005 14H4.8v-4H5a1.7 1.7 0 001.6-1 1.7 1.7 0 00-.3-1.9L6.2 7 8 5.2l.1.1A1.7 1.7 0 0010 5.6a1.7 1.7 0 001-1.6v-.2h4V4a1.7 1.7 0 001 1.6 1.7 1.7 0 001.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 00-.3 1.9 1.7 1.7 0 001.6 1h.2v4H21a1.7 1.7 0 00-1.6 1z"/></svg>
            Settings
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="status-dot"></div>
          <div><strong>System Online</strong><span>API connected</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">COMPUTER VISION WORKSPACE</p>
            <h1>Object Detection</h1>
          </div>
          <div className="topbar-actions">
            <span className="model-badge"><span></span> YOLOv8 Model</span>
            <button className="icon-button" type="button" aria-label="Notifications">
              <svg viewBox="0 0 24 24"><path d="M18 8a6 6 0 00-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></svg>
            </button>
          </div>
        </header>

        <div className="workspace">
          <section className="detection-card">
            <div className="card-toolbar">
              <div className="segmented-control" role="tablist" aria-label="Detection source">
                <button className={mode === "upload" ? "active" : ""} type="button" role="tab" aria-selected={mode === "upload"} onClick={() => { stopWebcam(); setMode("upload"); }}>
                  <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7 9m5-5l5 5"/><path d="M4 15v4h16v-4"/></svg>
                  Upload Image
                </button>
                <button className={mode === "webcam" ? "active" : ""} type="button" role="tab" aria-selected={mode === "webcam"} onClick={() => setMode("webcam")}>
                  <svg viewBox="0 0 24 24"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/></svg>
                  Live Webcam
                </button>
              </div>

              <label className="confidence-control">
                <span>Confidence</span>
                <strong>{(conf * 100).toFixed(0)}%</strong>
                <input type="range" min="0.1" max="0.95" step="0.05" value={conf} onChange={(e) => setConf(parseFloat(e.target.value))} />
              </label>
            </div>

            {error && <div className="error-banner"><span>!</span>{error}</div>}

            {mode === "upload" ? (
              <div className="preview-area">
                <label className="upload-overlay">
                  <input type="file" accept="image/*" onChange={handleFileUpload} />
                  <svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7 9m5-5l5 5"/><path d="M4 15v4h16v-4"/></svg>
                  <span>Choose an image</span>
                </label>
                <img ref={imgRef} alt="Detection preview" />
                <canvas ref={canvasRef} className="detection-canvas" />
              </div>
            ) : (
              <div className="preview-area webcam-area">
                <video ref={videoRef} autoPlay playsInline muted />
                <canvas ref={canvasRef} className="detection-canvas" />
                {!webcamActive && (
                  <div className="camera-placeholder">
                    <div className="camera-icon">
                      <svg viewBox="0 0 24 24"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/></svg>
                    </div>
                    <strong>Camera is inactive</strong>
                    <span>Start the camera to begin live detection</span>
                  </div>
                )}
                <button className={webcamActive ? "camera-button stop" : "camera-button"} type="button" onClick={webcamActive ? stopWebcam : startWebcam}>
                  <span></span>{webcamActive ? "Stop Camera" : "Start Camera"}
                </button>
              </div>
            )}
          </section>

          <aside className="results-card">
            <div className="results-heading">
              <div>
                <p>ANALYSIS</p>
                <h2>Detection Results</h2>
              </div>
              <span className={`status-pill ${detections.length ? "detected" : ""}`}>{detections.length ? "Detected" : "Idle"}</span>
            </div>

            <div className="result-summary">
              {loading ? (
                <div className="scanning-state"><div className="spinner"></div><strong>Analyzing image</strong><span>Running inference...</span></div>
              ) : (
                <>
                  <div className="summary-icon">
                    {detections.length > 0 ? (
                      <svg viewBox="0 0 24 24"><path d="M5 12l4 4L19 6"/></svg>
                    ) : (
                      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>
                    )}
                  </div>
                  <strong>{detections.length > 0 ? "Target Identified" : "No Target Detected"}</strong>
                  <span>{detections.length > 0 ? `${detections.length} object${detections.length > 1 ? "s" : ""} found in image` : "Upload or scan an image to begin"}</span>
                </>
              )}
            </div>

            {detections.length > 0 && (
              <div className="detections-list">
                {detections.map((det, index) => (
                  <div className="detection-row" key={`${det.label}-${index}`}>
                    <div className="detection-index">{index + 1}</div>
                    <div><strong>{det.label}</strong><span>Confidence</span></div>
                    <b>{(det.confidence * 100).toFixed(1)}%</b>
                  </div>
                ))}
              </div>
            )}

            <div className="metrics">
              <div><span>Inferences</span><strong>{detections.length}</strong></div>
              <div><span>Latency</span><strong>{inferenceMs !== null ? `${inferenceMs} ms` : "--"}</strong></div>
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}