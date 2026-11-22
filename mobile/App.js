import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Platform,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";

const TOP_INSET = Platform.OS === "android" ? (StatusBar.currentHeight || 28) + 8 : 14;
const BOTTOM_INSET = Platform.OS === "android" ? 36 : 14;

export default function App() {
  // Navigation & Settings
  const [activeTab, setActiveTab] = useState("detection"); // "detection" | "history" | "settings"
  const [sourceMode, setSourceMode] = useState("upload"); // "upload" | "camera"
  const [apiBase, setApiBase] = useState("https://custom-object-detection-recognition.onrender.com");
  const [conf, setConf] = useState(0.5);
  const [boxColor, setBoxColor] = useState("#10b981");

  // Backend Health
  const [apiOnline, setApiOnline] = useState(false);
  const [modelClasses, setModelClasses] = useState(["Samuel"]);

  // Camera & Image State
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState("front");
  const [capturedUri, setCapturedUri] = useState(null);
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });

  // Inference & History State
  const [detections, setDetections] = useState([]);
  const [inferenceMs, setInferenceMs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const cameraRef = useRef(null);

  const checkBackendHealth = async (baseUrl = apiBase) => {
    try {
      const res = await fetch(`${baseUrl}/health`);
      const data = await res.json();
      if (data.status === "ok") {
        setApiOnline(true);
        setModelClasses(Object.values(data.classes || {}));
        setError("");
        return;
      }
    } catch {
      // Handled below
    }
    setApiOnline(false);
  };

  useEffect(() => {
    checkBackendHealth();
  }, []);

  // Reliable multipart upload helper for Android & iOS
  const uploadFormData = (url, formData) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.timeout = 25000;

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error("Invalid JSON response from Flask server."));
          }
        } else {
          reject(new Error(`Server error (HTTP ${xhr.status}): ${xhr.responseText}`));
        }
      };

      xhr.onerror = () => {
        reject(new Error(`Upload failed to ${url}. Check Wi-Fi and Windows Firewall.`));
      };

      xhr.ontimeout = () => {
        reject(new Error("Prediction timed out after 25s."));
      };

      xhr.send(formData);
    });

  const runInference = async (uri, origW, origH, sourceLabel, overrideConf = conf) => {
    setLoading(true);
    setError("");

    const normalizedUri =
      Platform.OS === "android" &&
      !uri.startsWith("file://") &&
      !uri.startsWith("content://")
        ? `file://${uri}`
        : uri;

    const formData = new FormData();
    formData.append("image", {
      uri: normalizedUri,
      name: "mobile_frame.jpg",
      type: "image/jpeg",
    });
    formData.append("conf", String(overrideConf));

    try {
      const data = await uploadFormData(`${apiBase}/predict`, formData);

      if (data.detections) {
        setDetections(data.detections);
        setInferenceMs(data.inference_ms);
        setImageDims({
          width: data.image_width || origW || 1,
          height: data.image_height || origH || 1,
        });

        if (data.detections.length > 0) {
          const top = data.detections[0];
          setHistory((prev) => [
            {
              id: `${Date.now()}-${Math.random()}`,
              label: top.label,
              confidence: (top.confidence * 100).toFixed(1),
              count: data.detections.length,
              latency: data.inference_ms,
              source: sourceLabel,
              time: new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              }),
              uri: normalizedUri,
            },
            ...prev.slice(0, 29),
          ]);
        }
      }
    } catch (err) {
      setError(err.message || `Cannot reach Flask API at ${apiBase}.`);
    } finally {
      setLoading(false);
    }
  };

  const pickImageFromGallery = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.75,
    });

    if (!result.canceled && result.assets?.[0]) {
      const asset = result.assets[0];
      setDetections([]);
      setCapturedUri(asset.uri);
      setImageDims({ width: asset.width || 1, height: asset.height || 1 });
      await runInference(asset.uri, asset.width, asset.height, "Gallery Upload");
    }
  };

  const takeCameraShot = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.75 });
      if (photo?.uri) {
        setDetections([]);
        setCapturedUri(photo.uri);
        setImageDims({ width: photo.width || 1, height: photo.height || 1 });
        await runInference(photo.uri, photo.width, photo.height, "Camera Shot");
      }
    } catch {
      setError("Failed to capture photo from camera.");
    }
  };

  const retakeCameraShot = () => {
    setCapturedUri(null);
    setDetections([]);
    setInferenceMs(null);
    setError("");
  };

  const adjustConfidence = (delta) => {
    const nextConf = Math.min(0.95, Math.max(0.1, Number((conf + delta).toFixed(2))));
    setConf(nextConf);
    if (capturedUri) {
      runInference(
        capturedUri,
        imageDims.width,
        imageDims.height,
        sourceMode === "upload" ? "Gallery Upload" : "Camera Shot",
        nextConf
      );
    }
  };

  const aspectRatio =
    imageDims.width && imageDims.height ? imageDims.width / imageDims.height : 4 / 3;

  const renderBoundingBoxes = () => {
    if (!displaySize.width || !displaySize.height || detections.length === 0) return null;

    const scaleX = displaySize.width / imageDims.width;
    const scaleY = displaySize.height / imageDims.height;

    return detections.map((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox;
      const left = x1 * scaleX;
      const top = y1 * scaleY;
      const width = (x2 - x1) * scaleX;
      const height = (y2 - y1) * scaleY;

      return (
        <View
          key={`${det.label}-${idx}`}
          style={[
            styles.bbox,
            {
              left,
              top,
              width,
              height,
              borderColor: boxColor,
            },
          ]}
        >
          <View style={[styles.bboxBadge, { backgroundColor: boxColor }]}>
            <Text style={styles.bboxText}>
              {det.label} {(det.confidence * 100).toFixed(1)}%
            </Text>
          </View>
        </View>
      );
    });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor="#0b1324" />

      {/* Top Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.eyebrow}>VISIONLAB MOBILE</Text>
          <Text style={styles.headerTitle}>Object Detection</Text>
        </View>

        <TouchableOpacity
          style={[styles.statusBadge, apiOnline ? styles.badgeOnline : styles.badgeOffline]}
          onPress={() => checkBackendHealth()}
        >
          <View
            style={[
              styles.statusDot,
              { backgroundColor: apiOnline ? "#10b981" : "#ef4444" },
            ]}
          />
          <Text style={styles.statusText}>
            {apiOnline ? "YOLO26n Online" : "API Offline"}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Main Scroll Area */}
      <ScrollView contentContainerStyle={styles.scrollBody}>
        {/* TAB 1: DETECTION */}
        {activeTab === "detection" && (
          <>
            {/* Mode Switcher */}
            <View style={styles.segmentedRow}>
              <TouchableOpacity
                style={[styles.segmentBtn, sourceMode === "upload" && styles.segmentActive]}
                onPress={() => {
                  setSourceMode("upload");
                  setCapturedUri(null);
                  setDetections([]);
                  setError("");
                }}
              >
                <Text
                  style={[
                    styles.segmentText,
                    sourceMode === "upload" && styles.segmentTextActive,
                  ]}
                >
                  Upload Image
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.segmentBtn, sourceMode === "camera" && styles.segmentActive]}
                onPress={async () => {
                  setSourceMode("camera");
                  setCapturedUri(null);
                  setDetections([]);
                  setError("");
                  if (!permission?.granted) {
                    await requestPermission();
                  }
                }}
              >
                <Text
                  style={[
                    styles.segmentText,
                    sourceMode === "camera" && styles.segmentTextActive,
                  ]}
                >
                  Live Camera
                </Text>
              </TouchableOpacity>
            </View>

            {/* Confidence Stepper */}
            <View style={styles.card}>
              <View style={styles.confRow}>
                <Text style={styles.cardLabel}>Confidence Threshold</Text>
                <View style={styles.confControls}>
                  <TouchableOpacity
                    style={styles.stepBtn}
                    onPress={() => adjustConfidence(-0.05)}
                  >
                    <Text style={styles.stepBtnText}>-</Text>
                  </TouchableOpacity>
                  <Text style={styles.confValue}>{(conf * 100).toFixed(0)}%</Text>
                  <TouchableOpacity
                    style={styles.stepBtn}
                    onPress={() => adjustConfidence(0.05)}
                  >
                    <Text style={styles.stepBtnText}>+</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>

            {error ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            ) : null}

            {/* Stage Card */}
            <View style={styles.card}>
              {sourceMode === "upload" ? (
                <View style={styles.stageCenter}>
                  {capturedUri ? (
                    <View
                      style={[styles.mediaFrame, { aspectRatio }]}
                      onLayout={(e) => setDisplaySize(e.nativeEvent.layout)}
                    >
                      <Image
                        source={{ uri: capturedUri }}
                        style={styles.previewImage}
                        resizeMode="contain"
                      />
                      {renderBoundingBoxes()}
                    </View>
                  ) : (
                    <View style={styles.placeholderBox}>
                      <Text style={styles.placeholderTitle}>No Image Selected</Text>
                      <Text style={styles.placeholderSub}>
                        Choose a photo from your phone gallery to run YOLO detection
                      </Text>
                    </View>
                  )}

                  <TouchableOpacity style={styles.primaryBtn} onPress={pickImageFromGallery}>
                    <Text style={styles.primaryBtnText}>
                      {capturedUri ? "Choose Another Image" : "Select Image from Gallery"}
                    </Text>
                  </TouchableOpacity>
                </View>
              ) : (
                <View style={styles.stageCenter}>
                  {!permission?.granted ? (
                    <View style={styles.placeholderBox}>
                      <Text style={styles.placeholderTitle}>Camera Permission Needed</Text>
                      <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
                        <Text style={styles.primaryBtnText}>Grant Camera Access</Text>
                      </TouchableOpacity>
                    </View>
                  ) : capturedUri ? (
                    <>
                      <View
                        style={[styles.mediaFrame, { aspectRatio }]}
                        onLayout={(e) => setDisplaySize(e.nativeEvent.layout)}
                      >
                        <Image
                          source={{ uri: capturedUri }}
                          style={styles.previewImage}
                          resizeMode="contain"
                        />
                        {renderBoundingBoxes()}
                      </View>
                      <TouchableOpacity style={styles.secondaryBtn} onPress={retakeCameraShot}>
                        <Text style={styles.secondaryBtnText}>Retake Shot</Text>
                      </TouchableOpacity>
                    </>
                  ) : (
                    <>
                      <View style={styles.cameraContainer}>
                        <CameraView ref={cameraRef} style={styles.cameraView} facing={facing} />
                      </View>
                      <View style={styles.cameraActionRow}>
                        <TouchableOpacity
                          style={styles.secondaryBtn}
                          onPress={() => setFacing((f) => (f === "front" ? "back" : "front"))}
                        >
                          <Text style={styles.secondaryBtnText}>Flip Camera</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.primaryBtn} onPress={takeCameraShot}>
                          <Text style={styles.primaryBtnText}>Take Shot &amp; Detect</Text>
                        </TouchableOpacity>
                      </View>
                    </>
                  )}
                </View>
              )}
            </View>

            {/* Detection Results Card */}
            <View style={styles.card}>
              <View style={styles.resultsHeader}>
                <Text style={styles.sectionTitle}>Detection Results</Text>
                <View
                  style={[
                    styles.pill,
                    detections.length > 0 ? styles.pillDetected : styles.pillIdle,
                  ]}
                >
                  <Text
                    style={[
                      styles.pillText,
                      detections.length > 0 ? styles.pillTextDetected : styles.pillTextIdle,
                    ]}
                  >
                    {detections.length > 0 ? "DETECTED" : "IDLE"}
                  </Text>
                </View>
              </View>

              {loading ? (
                <View style={styles.loadingRow}>
                  <ActivityIndicator color="#10b981" />
                  <Text style={styles.loadingText}>Running YOLO26n inference...</Text>
                </View>
              ) : (
                <Text style={styles.verdictText}>
                  {detections.length > 0
                    ? `Target Identified (${detections.length} match)`
                    : "No Target Detected"}
                </Text>
              )}

              {detections.map((det, idx) => (
                <View key={`${det.label}-${idx}`} style={styles.detectionRow}>
                  <View style={styles.indexBadge}>
                    <Text style={styles.indexText}>{idx + 1}</Text>
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.detLabel}>{det.label}</Text>
                    <Text style={styles.detSub}>Confidence</Text>
                  </View>
                  <Text style={styles.detScore}>{(det.confidence * 100).toFixed(1)}%</Text>
                </View>
              ))}

              <View style={styles.metricsRow}>
                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>Inferences</Text>
                  <Text style={styles.metricValue}>{detections.length}</Text>
                </View>
                <View style={styles.metricTile}>
                  <Text style={styles.metricLabel}>Latency</Text>
                  <Text style={styles.metricValue}>
                    {inferenceMs !== null ? `${inferenceMs} ms` : "--"}
                  </Text>
                </View>
              </View>
            </View>
          </>
        )}

        {/* TAB 2: HISTORY */}
        {activeTab === "history" && (
          <View style={styles.card}>
            <View style={styles.resultsHeader}>
              <Text style={styles.sectionTitle}>Detection History ({history.length})</Text>
              {history.length > 0 && (
                <TouchableOpacity onPress={() => setHistory([])}>
                  <Text style={{ color: "#ef4444", fontWeight: "700" }}>Clear</Text>
                </TouchableOpacity>
              )}
            </View>

            {history.length === 0 ? (
              <Text style={styles.placeholderSub}>
                No detections logged yet. Upload an image or take a camera shot first.
              </Text>
            ) : (
              history.map((item) => (
                <View key={item.id} style={styles.historyRow}>
                  <Image source={{ uri: item.uri }} style={styles.historyThumb} />
                  <View style={{ flex: 1 }}>
                    <Text style={styles.detLabel}>{item.label}</Text>
                    <Text style={styles.detSub}>
                      {item.source} • {item.time} • {item.latency} ms
                    </Text>
                  </View>
                  <Text style={styles.detScore}>{item.confidence}%</Text>
                </View>
              ))
            )}
          </View>
        )}

        {/* TAB 3: SETTINGS */}
        {activeTab === "settings" && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Server &amp; Display Settings</Text>

            <Text style={[styles.cardLabel, { marginTop: 14 }]}>
              Flask Backend Wi-Fi URL
            </Text>
            <TextInput
              style={styles.input}
              value={apiBase}
              onChangeText={setApiBase}
              autoCapitalize="none"
              autoCorrect={false}
              placeholder="http://10.199.85.139:5000"
            />
            <TouchableOpacity
              style={[styles.primaryBtn, { marginTop: 10 }]}
              onPress={() => checkBackendHealth(apiBase)}
            >
              <Text style={styles.primaryBtnText}>Test Connection</Text>
            </TouchableOpacity>

            <Text style={[styles.cardLabel, { marginTop: 18 }]}>
              Loaded Classes: {modelClasses.join(", ") || "None"}
            </Text>

            <Text style={[styles.cardLabel, { marginTop: 18 }]}>Bounding Box Color</Text>
            <View style={styles.colorRow}>
              {[
                { name: "Emerald", hex: "#10b981" },
                { name: "Sky", hex: "#0ea5e9" },
                { name: "Amber", hex: "#f59e0b" },
                { name: "Rose", hex: "#f43f5e" },
              ].map((c) => (
                <TouchableOpacity
                  key={c.hex}
                  style={[
                    styles.colorChip,
                    boxColor === c.hex && { borderColor: c.hex, borderWidth: 2 },
                  ]}
                  onPress={() => setBoxColor(c.hex)}
                >
                  <View style={[styles.colorDot, { backgroundColor: c.hex }]} />
                  <Text style={styles.colorName}>{c.name}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        )}
      </ScrollView>

      {/* Bottom Tab Navigation */}
      <View style={styles.bottomNav}>
        {[
          { key: "detection", label: "Detection" },
          { key: "history", label: `History (${history.length})` },
          { key: "settings", label: "Settings" },
        ].map((tab) => (
          <TouchableOpacity
            key={tab.key}
            style={[styles.navTab, activeTab === tab.key && styles.navTabActive]}
            onPress={() => setActiveTab(tab.key)}
          >
            <Text
              style={[
                styles.navTabText,
                activeTab === tab.key && styles.navTabTextActive,
              ]}
            >
              {tab.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f1f5f9" },
  header: {
    backgroundColor: "#0b1324",
    paddingHorizontal: 18,
    paddingTop: TOP_INSET,
    paddingBottom: 14,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  eyebrow: { color: "#10b981", fontSize: 10, fontWeight: "800", letterSpacing: 1 },
  headerTitle: { color: "#ffffff", fontSize: 20, fontWeight: "800" },
  statusBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    gap: 6,
  },
  badgeOnline: { backgroundColor: "rgba(16, 185, 129, 0.16)" },
  badgeOffline: { backgroundColor: "rgba(239, 68, 68, 0.2)" },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
  statusText: { color: "#ffffff", fontSize: 12, fontWeight: "700" },
  scrollBody: { padding: 14, paddingBottom: 32, gap: 12 },
  segmentedRow: {
    flexDirection: "row",
    backgroundColor: "#e2e8f0",
    borderRadius: 12,
    padding: 4,
  },
  segmentBtn: { flex: 1, paddingVertical: 10, alignItems: "center", borderRadius: 9 },
  segmentActive: { backgroundColor: "#ffffff" },
  segmentText: { fontSize: 14, fontWeight: "600", color: "#475569" },
  segmentTextActive: { color: "#0f172a", fontWeight: "700" },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  cardLabel: { fontSize: 13, fontWeight: "700", color: "#334155" },
  confRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  confControls: { flexDirection: "row", alignItems: "center", gap: 12 },
  stepBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: "#f1f5f9",
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "#cbd5e1",
  },
  stepBtnText: { fontSize: 18, fontWeight: "700", color: "#0f172a" },
  confValue: { fontSize: 15, fontWeight: "800", color: "#10b981", minWidth: 42, textAlign: "center" },
  errorBox: {
    backgroundColor: "#fef2f2",
    borderColor: "#fecaca",
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
  },
  errorText: { color: "#dc2626", fontSize: 13, fontWeight: "600" },
  stageCenter: { alignItems: "center", gap: 12 },
  placeholderBox: {
    width: "100%",
    paddingVertical: 36,
    paddingHorizontal: 16,
    backgroundColor: "#f8fafc",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderStyle: "dashed",
    alignItems: "center",
    gap: 6,
  },
  placeholderTitle: { fontSize: 16, fontWeight: "700", color: "#0f172a" },
  placeholderSub: { fontSize: 13, color: "#64748b", textAlign: "center" },
  mediaFrame: {
    width: "100%",
    position: "relative",
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#0f172a",
  },
  previewImage: { width: "100%", height: "100%" },
  bbox: {
    position: "absolute",
    borderWidth: 2.5,
    borderRadius: 4,
  },
  bboxBadge: {
    position: "absolute",
    top: -22,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  bboxText: { color: "#ffffff", fontSize: 11, fontWeight: "800" },
  cameraContainer: {
    width: "100%",
    height: 360,
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#000000",
  },
  cameraView: { flex: 1 },
  cameraActionRow: { flexDirection: "row", gap: 10, width: "100%" },
  primaryBtn: {
    flex: 1,
    backgroundColor: "#10b981",
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    alignItems: "center",
  },
  primaryBtnText: { color: "#ffffff", fontWeight: "700", fontSize: 14 },
  secondaryBtn: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 10,
    backgroundColor: "#e2e8f0",
    alignItems: "center",
  },
  secondaryBtnText: { color: "#0f172a", fontWeight: "700", fontSize: 14 },
  resultsHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  sectionTitle: { fontSize: 16, fontWeight: "800", color: "#0f172a" },
  pill: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 999 },
  pillDetected: { backgroundColor: "#dcfce7" },
  pillIdle: { backgroundColor: "#f1f5f9" },
  pillText: { fontSize: 11, fontWeight: "800" },
  pillTextDetected: { color: "#15803d" },
  pillTextIdle: { color: "#64748b" },
  loadingRow: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 8 },
  loadingText: { color: "#475569", fontSize: 14, fontWeight: "600" },
  verdictText: { fontSize: 15, fontWeight: "700", color: "#0f172a", marginBottom: 10 },
  detectionRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    padding: 10,
    backgroundColor: "#f8fafc",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#e2e8f0",
    marginBottom: 8,
  },
  indexBadge: {
    width: 26,
    height: 26,
    borderRadius: 13,
    backgroundColor: "#10b981",
    alignItems: "center",
    justifyContent: "center",
  },
  indexText: { color: "#ffffff", fontWeight: "800", fontSize: 12 },
  detLabel: { fontSize: 15, fontWeight: "700", color: "#0f172a" },
  detSub: { fontSize: 12, color: "#64748b" },
  detScore: { fontSize: 15, fontWeight: "800", color: "#10b981" },
  metricsRow: { flexDirection: "row", gap: 10, marginTop: 4 },
  metricTile: {
    flex: 1,
    backgroundColor: "#f8fafc",
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  metricLabel: { fontSize: 12, color: "#64748b" },
  metricValue: { fontSize: 16, fontWeight: "800", color: "#0f172a", marginTop: 2 },
  historyRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#e2e8f0",
  },
  historyThumb: { width: 56, height: 42, borderRadius: 6, backgroundColor: "#0f172a" },
  input: {
    marginTop: 6,
    borderWidth: 1,
    borderColor: "#cbd5e1",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    backgroundColor: "#f8fafc",
  },
  colorRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 8 },
  colorChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#cbd5e1",
  },
  colorDot: { width: 10, height: 10, borderRadius: 5 },
  colorName: { fontSize: 13, fontWeight: "600", color: "#0f172a" },
  bottomNav: {
    flexDirection: "row",
    backgroundColor: "#0b1324",
    paddingTop: 10,
    paddingBottom: BOTTOM_INSET,
    paddingHorizontal: 10,
    gap: 6,
  },
  navTab: { flex: 1, paddingVertical: 10, borderRadius: 10, alignItems: "center" },
  navTabActive: { backgroundColor: "#1e293b" },
  navTabText: { color: "#94a3b8", fontWeight: "700", fontSize: 13 },
  navTabTextActive: { color: "#10b981" },
});