import { useEffect, useRef, useState } from "react";
import "./App.css";

export default function App() {
  // Navigation & Popovers
  const [activeTab, setActiveTab] = useState("detection"); // "detection" | "history" | "settings"
  const [showNotifications, setShowNotifications] = useState(false);
  const [showModelInfo, setShowModelInfo] = useState(false);

  // Settings State
  const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:5000" : "https://custom-object-detection-recognition.onrender.com"));
  const [boxColor, setBoxColor] = useState("#10b981");
  const [webcamAutoDetect, setWebcamAutoDetect] = useState(false);
  const [conf, setConf] = useState(0.5);

  // Backend Health State
  const [systemHealth, setSystemHealth] = useState({
    online: false,
    classes: ["Samuel"],
    lastChecked: null,
  });

  // Detection & Media State
  const [mode, setMode] = useState("upload"); // "upload" | "webcam"
  const [detections, setDetections] = useState([]);
  const [inferenceMs, setInferenceMs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [webcamActive, setWebcamActive] = useState(false);
  const [shotTaken, setShotTaken] = useState(false);
  const [error, setError] = useState("");

  // History & Notifications State
  const [history, setHistory] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const confRef = useRef(conf);
  const boxColorRef = useRef(boxColor);
  const lastUploadFileRef = useRef(null);
  const lastShotRef = useRef(null);

  useEffect(() => {
    confRef.current = conf;
  }, [conf]);

  useEffect(() => {
    boxColorRef.current = boxColor;
  }, [boxColor]);

  // 1. Live Backend Health Check
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${apiBase}/health`);
      const data = await res.json();
      if (data.status === "ok") {
        setSystemHealth({
          online: true,
          classes: Object.values(data.classes || {}),
          lastChecked: new Date().toLocaleTimeString(),
        });
        return;
      }
    } catch {
      // Handled below
    }
    setSystemHealth((prev) => ({
      ...prev,
      online: false,
      lastChecked: new Date().toLocaleTimeString(),
    }));
  };

  useEffect(() => {
    checkBackendHealth();
  }, [apiBase]);

  const clearOverlay = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const createThumbnail = (sourceEl) => {
    try {
      const thumb = document.createElement("canvas");
      thumb.width = 96;
      thumb.height = 72;
      thumb.getContext("2d").drawImage(sourceEl, 0, 0, 96, 72);
      return thumb.toDataURL("image/jpeg", 0.7);
    } catch {
      return null;
    }
  };

  const drawBoxes = (boxes, origW, origH, displayW, displayH) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = displayW;
    canvas.height = displayH;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    const scaleX = displayW / origW;
    const scaleY = displayH / origH;
    const color = boxColorRef.current;

    boxes.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const sx = x1 * scaleX;
      const sy = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(sx, sy, sw, sh);

      const text = `${det.label} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.font = "bold 14px sans-serif";
      const textWidth = ctx.measureText(text).width;
      const badgeY = sy > 26 ? sy - 24 : sy;

      ctx.fillStyle = color;
      ctx.fillRect(sx, badgeY, textWidth + 12, 24);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, sx + 6, badgeY + 17);
    });
  };

  const recordDetectionEvent = (dets, latency, sourceType, sourceEl, logToHistory) => {
    if (!logToHistory || dets.length === 0) return;

    const top = dets[0];
    const time = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    const thumbnail = sourceEl ? createThumbnail(sourceEl) : null;

    setHistory((prev) => [
      {
        id: `${Date.now()}-${Math.random()}`,
        label: top.label,
        confidence: (top.confidence * 100).toFixed(1),
        count: dets.length,
        latency,
        source: sourceType,
        time,
        thumbnail,
      },
      ...prev.slice(0, 29),
    ]);

    setNotifications((prev) => [
      {
        id: `${Date.now()}-${Math.random()}`,
        title: `Target Identified: ${top.label}`,
        detail: `${(top.confidence * 100).toFixed(1)}% confidence via ${sourceType}`,
        time,
      },
      ...prev.slice(0, 14),
    ]);
    setUnreadCount((c) => c + 1);
  };

  const runInference = async (blob, displayW, displayH, options = {}) => {
    const { overrideConf, sourceType = "Upload", sourceEl = null, logEvent = false } = options;

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    formData.append("conf", overrideConf ?? confRef.current);

    try {
      setError("");
      const res = await fetch(`${apiBase}/predict`, { method: "POST", body: formData });
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
        recordDetectionEvent(data.detections, data.inference_ms, sourceType, sourceEl, logEvent);
      }
    } catch {
      setError(`Cannot reach Flask backend at ${apiBase}. Make sure backend/app.py is running.`);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    lastUploadFileRef.current = file;
    setDetections([]);
    clearOverlay();
    setLoading(true);

    const url = URL.createObjectURL(file);
    const img = imgRef.current;
    img.src = url;

    img.onload = async () => {
      await runInference(file, img.clientWidth, img.clientHeight, {
        sourceType: "Image Upload",
        sourceEl: img,
        logEvent: true,
      });
      setLoading(false);
    };
  };

  const startLiveLoop = () => {
    clearInterval(intervalRef.current);
    if (!webcamAutoDetect) return;

    const captureCanvas = document.createElement("canvas");
    intervalRef.current = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.paused || video.readyState < 2) return;

      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      captureCanvas.getContext("2d").drawImage(video, 0, 0);

      captureCanvas.toBlob(
        (blob) => {
          if (blob) {
            runInference(blob, video.clientWidth, video.clientHeight, {
              sourceType: "Live Stream",
              logEvent: false,
            });
          }
        },
        "image/jpeg",
        0.8
      );
    }, 350);
  };

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setWebcamActive(true);
      setShotTaken(false);
      lastShotRef.current = null;
      startLiveLoop();
    } catch {
      setError("Unable to access webcam. Check browser camera permissions.");
    }
  };

  const takeShot = () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    clearInterval(intervalRef.current);
    video.pause();
    setShotTaken(true);
    setLoading(true);

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCanvas.getContext("2d").drawImage(video, 0, 0);

    captureCanvas.toBlob(
      async (blob) => {
        if (blob) {
          lastShotRef.current = blob;
          await runInference(blob, video.clientWidth, video.clientHeight, {
            sourceType: "Webcam Shot",
            sourceEl: video,
            logEvent: true,
          });
        }
        setLoading(false);
      },
      "image/jpeg",
      0.92
    );
  };

  const retakeShot = () => {
    const video = videoRef.current;
    clearOverlay();
    lastShotRef.current = null;
    setDetections([]);
    setInferenceMs(null);
    setShotTaken(false);
    if (video) {
      video.play();
      startLiveLoop();
    }
  };

  const stopWebcam = () => {
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    clearOverlay();
    lastShotRef.current = null;
    setWebcamActive(false);
    setShotTaken(false);
    setDetections([]);
    setInferenceMs(null);
  };

  const handleConfChange = (e) => {
    const newConf = parseFloat(e.target.value);
    setConf(newConf);
    confRef.current = newConf;

    if (mode === "upload" && lastUploadFileRef.current && imgRef.current) {
      runInference(
        lastUploadFileRef.current,
        imgRef.current.clientWidth,
        imgRef.current.clientHeight,
        { overrideConf: newConf }
      );
    } else if (mode === "webcam" && shotTaken && lastShotRef.current && videoRef.current) {
      runInference(
        lastShotRef.current,
        videoRef.current.clientWidth,
        videoRef.current.clientHeight,
        { overrideConf: newConf }
      );
    }
  };

  useEffect(() => () => stopWebcam(), []);

  return (
    <div className="app-shell">
      {/* Left Navigation Sidebar */}
      <aside className="sidebar">
        <div
          className="brand"
          style={{ cursor: "pointer" }}
          onClick={() => setActiveTab("detection")}
          title="Return to Detection Workspace"
        >
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 3l7 3v5c0 4.6-2.9 7.9-7 10-4.1-2.1-7-5.4-7-10V6l7-3z" />
              <circle cx="12" cy="11" r="3" />
            </svg>
          </div>
          <span>Vision<span>Lab</span></span>
        </div>

        <nav className="side-nav" aria-label="Primary navigation">
          <button
            className={activeTab === "detection" ? "nav-item active" : "nav-item"}
            type="button"
            title="Object & Face Detection"
            onClick={() => setActiveTab("detection")}
          >
            <svg viewBox="0 0 24 24">
              <rect x="3" y="3" width="7" height="7" rx="2" />
              <rect x="14" y="3" width="7" height="7" rx="2" />
              <rect x="3" y="14" width="7" height="7" rx="2" />
              <rect x="14" y="14" width="7" height="7" rx="2" />
            </svg>
            Detection
          </button>

          <button
            className={activeTab === "history" ? "nav-item active" : "nav-item"}
            type="button"
            title="Detection History Log"
            onClick={() => setActiveTab("history")}
          >
            <svg viewBox="0 0 24 24">
              <path d="M3 12a9 9 0 1018 0 9 9 0 10-18 0" />
              <path d="M12 7v5l3 2" />
            </svg>
            History {history.length > 0 && `(${history.length})`}
          </button>

          <button
            className={activeTab === "settings" ? "nav-item active" : "nav-item"}
            type="button"
            title="Model & Camera Settings"
            onClick={() => setActiveTab("settings")}
          >
            <svg viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.7 1.7 0 00.3 1.9l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 00-1.9-.3 1.7 1.7 0 00-1 1.6v.2H11V20a1.7 1.7 0 00-1-1.6 1.7 1.7 0 00-1.9.3l-.1.1L6.2 17l.1-.1a1.7 1.7 0 00.3-1.9A1.7 1.7 0 005 14H4.8v-4H5a1.7 1.7 0 001.6-1 1.7 1.7 0 00-.3-1.9L6.2 7 8 5.2l.1.1A1.7 1.7 0 0010 5.6a1.7 1.7 0 001-1.6v-.2h4V4a1.7 1.7 0 001 1.6 1.7 1.7 0 001.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 00-.3 1.9 1.7 1.7 0 001.6 1h.2v4H21a1.7 1.7 0 00-1.6 1z" />
            </svg>
            Settings
          </button>
        </nav>

        <div
          className="sidebar-footer"
          style={{ cursor: "pointer" }}
          onClick={checkBackendHealth}
          title="Click to re-check Flask API health"
        >
          <div
            className="status-dot"
            style={{ background: systemHealth.online ? "#10b981" : "#ef4444" }}
          />
          <div>
            <strong>{systemHealth.online ? "System Online" : "API Offline"}</strong>
            <span>{systemHealth.online ? "Flask connected" : "Click to retry"}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">COMPUTER VISION WORKSPACE</p>
            <h1>
              {activeTab === "detection" && "Object Detection"}
              {activeTab === "history" && "Detection History"}
              {activeTab === "settings" && "System Settings"}
            </h1>
          </div>

          <div className="topbar-actions" style={{ position: "relative" }}>
            <button
              type="button"
              className="model-badge"
              style={{ cursor: "pointer" }}
              onClick={() => {
                checkBackendHealth();
                setShowModelInfo((v) => !v);
                setShowNotifications(false);
              }}
              title="Click for model diagnostics"
            >
              <span style={{ background: systemHealth.online ? "#10b981" : "#ef4444" }} />
              YOLO26n Model
            </button>

            <button
              className="icon-button"
              type="button"
              aria-label="Notifications"
              style={{ position: "relative" }}
              onClick={() => {
                setShowNotifications((v) => !v);
                setShowModelInfo(false);
                setUnreadCount(0);
              }}
            >
              <svg viewBox="0 0 24 24">
                <path d="M18 8a6 6 0 00-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" />
              </svg>
              {unreadCount > 0 && (
                <span
                  style={{
                    position: "absolute",
                    top: 6,
                    right: 6,
                    width: 9,
                    height: 9,
                    borderRadius: "50%",
                    background: "#10b981",
                  }}
                />
              )}
            </button>

            {/* Model Info Popover */}
            {showModelInfo && (
              <div
                style={{
                  position: "absolute",
                  top: "115%",
                  right: 48,
                  width: 280,
                  background: "#ffffff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 12,
                  padding: "1rem",
                  boxShadow: "0 12px 28px rgba(15, 23, 42, 0.12)",
                  zIndex: 50,
                  fontSize: "0.85rem",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <strong>Model Diagnostics</strong>
                  <span style={{ color: systemHealth.online ? "#10b981" : "#ef4444", fontWeight: 700 }}>
                    {systemHealth.online ? "ONLINE" : "OFFLINE"}
                  </span>
                </div>
                <div style={{ color: "#475569", display: "grid", gap: 4 }}>
                  <div>Weights: <strong>backend/models/best.pt</strong></div>
                  <div>Classes: <strong>{systemHealth.classes.join(", ") || "None"}</strong></div>
                  <div>Endpoint: <strong>{apiBase}</strong></div>
                  {systemHealth.lastChecked && (
                    <div style={{ fontSize: "0.75rem", color: "#94a3b8", marginTop: 4 }}>
                      Checked at {systemHealth.lastChecked}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Notifications Popover */}
            {showNotifications && (
              <div
                style={{
                  position: "absolute",
                  top: "115%",
                  right: 0,
                  width: 310,
                  background: "#ffffff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 12,
                  padding: "1rem",
                  boxShadow: "0 12px 28px rgba(15, 23, 42, 0.12)",
                  zIndex: 50,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <strong style={{ fontSize: "0.9rem" }}>Detection Alerts</strong>
                  {notifications.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setNotifications([])}
                      style={{
                        border: "none",
                        background: "transparent",
                        color: "#10b981",
                        fontSize: "0.78rem",
                        fontWeight: 600,
                        cursor: "pointer",
                      }}
                    >
                      Clear all
                    </button>
                  )}
                </div>

                {notifications.length === 0 ? (
                  <p style={{ fontSize: "0.82rem", color: "#64748b", margin: 0 }}>
                    No detection alerts yet. Upload an image or take a webcam shot.
                  </p>
                ) : (
                  <div style={{ display: "grid", gap: 8, maxHeight: 240, overflowY: "auto" }}>
                    {notifications.map((n) => (
                      <div
                        key={n.id}
                        style={{
                          padding: "0.6rem",
                          borderRadius: 8,
                          background: "#f8fafc",
                          border: "1px solid #e2e8f0",
                          fontSize: "0.8rem",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <strong style={{ color: "#0f172a" }}>{n.title}</strong>
                          <span style={{ color: "#94a3b8", fontSize: "0.72rem" }}>{n.time}</span>
                        </div>
                        <div style={{ color: "#475569", marginTop: 2 }}>{n.detail}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </header>

        {/* TAB 1: DETECTION WORKSPACE */}
        {activeTab === "detection" && (
          <div className="workspace">
            <section className="detection-card">
              <div className="card-toolbar">
                <div className="segmented-control" role="tablist" aria-label="Detection source">
                  <button
                    className={mode === "upload" ? "active" : ""}
                    type="button"
                    role="tab"
                    aria-selected={mode === "upload"}
                    onClick={() => {
                      stopWebcam();
                      setMode("upload");
                    }}
                  >
                    <svg viewBox="0 0 24 24">
                      <path d="M12 16V4m0 0L7 9m5-5l5 5" />
                      <path d="M4 15v4h16v-4" />
                    </svg>
                    Upload Image
                  </button>
                  <button
                    className={mode === "webcam" ? "active" : ""}
                    type="button"
                    role="tab"
                    aria-selected={mode === "webcam"}
                    onClick={() => {
                      clearOverlay();
                      setDetections([]);
                      setMode("webcam");
                    }}
                  >
                    <svg viewBox="0 0 24 24">
                      <rect x="3" y="6" width="13" height="12" rx="2" />
                      <path d="M16 10l5-3v10l-5-3" />
                    </svg>
                    Live Webcam
                  </button>
                </div>

                <label className="confidence-control">
                  <span>Confidence</span>
                  <strong>{(conf * 100).toFixed(0)}%</strong>
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.05"
                    value={conf}
                    onChange={handleConfChange}
                  />
                </label>
              </div>

              {error && <div className="error-banner"><span>!</span>{error}</div>}

              {mode === "upload" ? (
                <div className="preview-area">
                  <label className="upload-overlay">
                    <input type="file" accept="image/*" onChange={handleFileUpload} />
                    <svg viewBox="0 0 24 24">
                      <path d="M12 16V4m0 0L7 9m5-5l5 5" />
                      <path d="M4 15v4h16v-4" />
                    </svg>
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
                        <svg viewBox="0 0 24 24">
                          <rect x="3" y="6" width="13" height="12" rx="2" />
                          <path d="M16 10l5-3v10l-5-3" />
                        </svg>
                      </div>
                      <strong>Camera is inactive</strong>
                      <span>Start the camera to take a shot or run live detection</span>
                    </div>
                  )}

                  <div
                    style={{
                      position: "absolute",
                      bottom: "1.25rem",
                      left: "50%",
                      transform: "translateX(-50%)",
                      display: "flex",
                      gap: "0.75rem",
                      zIndex: 10,
                    }}
                  >
                    {webcamActive && (
                      <button
                        className="camera-button"
                        style={{
                          position: "static",
                          transform: "none",
                          background: shotTaken ? "#2563eb" : "#10b981",
                          color: "#ffffff",
                        }}
                        type="button"
                        onClick={shotTaken ? retakeShot : takeShot}
                      >
                        {shotTaken ? "Retake Shot" : "Take Shot & Detect"}
                      </button>
                    )}
                    <button
                      className={webcamActive ? "camera-button stop" : "camera-button"}
                      style={{ position: "static", transform: "none" }}
                      type="button"
                      onClick={webcamActive ? stopWebcam : startWebcam}
                    >
                      <span></span>{webcamActive ? "Stop Camera" : "Start Camera"}
                    </button>
                  </div>
                </div>
              )}
            </section>

            <aside className="results-card">
              <div className="results-heading">
                <div>
                  <p>ANALYSIS</p>
                  <h2>Detection Results</h2>
                </div>
                <span className={`status-pill ${detections.length ? "detected" : ""}`}>
                  {detections.length ? "Detected" : "Idle"}
                </span>
              </div>

              <div className="result-summary">
                {loading ? (
                  <div className="scanning-state">
                    <div className="spinner"></div>
                    <strong>Analyzing frame</strong>
                    <span>Running inference...</span>
                  </div>
                ) : (
                  <>
                    <div className="summary-icon">
                      {detections.length > 0 ? (
                        <svg viewBox="0 0 24 24"><path d="M5 12l4 4L19 6" /></svg>
                      ) : (
                        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /><path d="M8 12h8" /></svg>
                      )}
                    </div>
                    <strong>{detections.length > 0 ? "Target Identified" : "No Target Detected"}</strong>
                    <span>
                      {detections.length > 0
                        ? `${detections.length} object${detections.length > 1 ? "s" : ""} found in frame`
                        : "Upload an image or take a webcam shot"}
                    </span>
                  </>
                )}
              </div>

              {detections.length > 0 && (
                <div className="detections-list">
                  {detections.map((det, index) => (
                    <div className="detection-row" key={`${det.label}-${index}`}>
                      <div className="detection-index">{index + 1}</div>
                      <div>
                        <strong>{det.label}</strong>
                        <span>Confidence</span>
                      </div>
                      <b>{(det.confidence * 100).toFixed(1)}%</b>
                    </div>
                  ))}
                </div>
              )}

              <div className="metrics">
                <div>
                  <span>Inferences</span>
                  <strong>{detections.length}</strong>
                </div>
                <div>
                  <span>Latency</span>
                  <strong>{inferenceMs !== null ? `${inferenceMs} ms` : "--"}</strong>
                </div>
              </div>
            </aside>
          </div>
        )}

        {/* TAB 2: HISTORY LOG */}
        {activeTab === "history" && (
          <section className="detection-card" style={{ padding: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
              <div>
                <h2 style={{ fontSize: "1.1rem", margin: 0 }}>Session Identification Log</h2>
                <p style={{ color: "#64748b", fontSize: "0.85rem", margin: "4px 0 0" }}>
                  Every confirmed detection from uploaded images and webcam snapshots
                </p>
              </div>
              {history.length > 0 && (
                <button
                  type="button"
                  onClick={() => setHistory([])}
                  style={{
                    padding: "0.5rem 1rem",
                    borderRadius: 8,
                    border: "1px solid #fecaca",
                    background: "#fef2f2",
                    color: "#dc2626",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  Clear History
                </button>
              )}
            </div>

            {history.length === 0 ? (
              <div style={{ textAlign: "center", padding: "3rem 1rem", color: "#64748b" }}>
                <strong>No detections recorded yet</strong>
                <p style={{ fontSize: "0.85rem", marginTop: 6 }}>
                  Run an image upload or take a webcam shot in the Detection tab to populate your log.
                </p>
              </div>
            ) : (
              <div style={{ display: "grid", gap: "0.75rem" }}>
                {history.map((item) => (
                  <div
                    key={item.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: "1rem",
                      padding: "0.85rem 1rem",
                      borderRadius: 10,
                      border: "1px solid #e2e8f0",
                      background: "#f8fafc",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.9rem" }}>
                      {item.thumbnail && (
                        <img
                          src={item.thumbnail}
                          alt={item.label}
                          style={{ width: 64, height: 48, objectFit: "cover", borderRadius: 6 }}
                        />
                      )}
                      <div>
                        <strong style={{ fontSize: "0.95rem", color: "#0f172a" }}>
                          {item.label} ({item.count} match)
                        </strong>
                        <div style={{ fontSize: "0.8rem", color: "#64748b" }}>
                          {item.source} • {item.time} • {item.latency} ms
                        </div>
                      </div>
                    </div>
                    <span
                      style={{
                        padding: "0.35rem 0.75rem",
                        borderRadius: 999,
                        background: "#dcfce7",
                        color: "#15803d",
                        fontWeight: 700,
                        fontSize: "0.85rem",
                      }}
                    >
                      {item.confidence}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* TAB 3: SETTINGS */}
        {activeTab === "settings" && (
          <section className="detection-card" style={{ padding: "1.5rem", maxWidth: 640 }}>
            <h2 style={{ fontSize: "1.1rem", marginBottom: "1.25rem" }}>Inference &amp; UI Configuration</h2>

            <div style={{ display: "grid", gap: "1.25rem" }}>
              <div>
                <label style={{ display: "block", fontWeight: 600, fontSize: "0.88rem", marginBottom: 6 }}>
                  Flask Backend API URL
                </label>
                <input
                  type="text"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "0.6rem 0.8rem",
                    borderRadius: 8,
                    border: "1px solid #cbd5e1",
                    fontSize: "0.9rem",
                  }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontWeight: 600, fontSize: "0.88rem", marginBottom: 6 }}>
                  Default Confidence Threshold: {(conf * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={conf}
                  onChange={(e) => setConf(parseFloat(e.target.value))}
                  style={{ width: "100%" }}
                />
              </div>

              <div>
                <label style={{ display: "block", fontWeight: 600, fontSize: "0.88rem", marginBottom: 6 }}>
                  Bounding Box Accent Color
                </label>
                <div style={{ display: "flex", gap: "0.6rem" }}>
                  {[
                    { name: "Emerald", hex: "#10b981" },
                    { name: "Sky Blue", hex: "#0ea5e9" },
                    { name: "Amber", hex: "#f59e0b" },
                    { name: "Rose", hex: "#f43f5e" },
                  ].map((c) => (
                    <button
                      key={c.hex}
                      type="button"
                      onClick={() => setBoxColor(c.hex)}
                      style={{
                        padding: "0.45rem 0.85rem",
                        borderRadius: 8,
                        border: boxColor === c.hex ? `2px solid ${c.hex}` : "1px solid #cbd5e1",
                        background: "#ffffff",
                        fontWeight: 600,
                        fontSize: "0.82rem",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <span style={{ width: 10, height: 10, borderRadius: "50%", background: c.hex }} />
                      {c.name}
                    </button>
                  ))}
                </div>
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "0.9rem",
                  borderRadius: 10,
                  background: "#f8fafc",
                  border: "1px solid #e2e8f0",
                }}
              >
                <div>
                  <strong style={{ fontSize: "0.9rem", display: "block" }}>
                    Continuous Webcam Auto-Tracking
                  </strong>
                  <span style={{ fontSize: "0.8rem", color: "#64748b" }}>
                    When off, webcam waits for you to click "Take Shot &amp; Detect"
                  </span>
                </div>
                <input
                  type="checkbox"
                  checked={webcamAutoDetect}
                  onChange={(e) => setWebcamAutoDetect(e.target.checked)}
                  style={{ width: 18, height: 18, cursor: "pointer" }}
                />
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}