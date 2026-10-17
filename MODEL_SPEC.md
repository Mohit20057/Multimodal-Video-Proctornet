# AI Model Integration Card: MultimodalVideoProctorNet

> **Instructions for the AI Assistant / Developer Integrating this Model:**
> You are integrating a production-ready, 36-layer Multimodal Video Vision Transformer into an application. Use the exact tensor contracts, file paths, pre-processing rules, and operational logic below.

---

## 1. Model Summary & Capabilities
- **Model Name:** `MultimodalVideoProctorNet`
- **Architecture:** 36 Hidden Layers (~33.9 Million Parameters):
  - 17-layer R(2+1)D ResNet Spatiotemporal Video Backbone
  - 4-layer Audio 2D-CNN Backbone (64-band Log-Mel)
  - 2 Linear Alignment Projections (to 384-dim)
  - 12-layer Deep Cross-Modal Transformer (6 Bidirectional Blocks, 8 heads, GELU FFN)
  - Dual Global Temporal Pooling (AvgPool $\oplus$ MaxPool = 768-dim)
  - 1 Dense Classification Head (768 $\rightarrow$ 256 $\rightarrow$ 4)
- **Primary Objective:** Real-time behavioral anomaly detection during technical interviews, coding assessments, and remote examinations.
- **Model Storage Locations (Absolute & Relative Paths):**
  - **Production ONNX Model (Recommended):**
    - Absolute Path: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\models\video_proctor_model.onnx`
    - Relative Path: `ml/models/video_proctor_model.onnx`
    - File Size: `136 MB` (136,159,548 bytes)
  - **PyTorch Weights Checkpoint:**
    - Absolute Path: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\models\best_proctor_model.pt`
    - Relative Path: `ml/models/best_proctor_model.pt`
    - File Size: `407 MB` (407,827,100 bytes)
  - **Dataset File:**
    - Absolute Path: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\data\proctoring_dataset.h5`
    - Relative Path: `ml/data/proctoring_dataset.h5`
    - File Size: `2.97 GB` (Chunked HDF5)
  - **Explainability & Metrics:**
    - Confusion Matrix: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\models\confusion_matrix.png`
    - Grad-CAM Heatmap: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\models\gradcam_explanation.png`
  - **SDK & API Implementations:**
    - Python SDK Wrapper: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\src\proctor_sdk.py`
    - FastAPI Microservice: `d:\PROJECTS\PROJECTS\Neural Newtork Model\ml\src\api.py`
    - Virtual Environment Python: `d:\PROJECTS\PROJECTS\Neural Newtork Model\.venv\Scripts\python.exe`

---

## 2. Input / Output Tensor Contracts

### Input Tensors (Dict keys passed to ONNX Runtime)

| Input Name | Shape | Data Type | Value Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`video`** | `[B, 16, 3, 112, 112]` | `float32` | `[-1.0, 1.0]` | Batch of 16 uniform temporal frames of face-centered RGB crops ($112 \times 112$). Normalized as `(pixel / 127.5) - 1.0`. Dimension order: `[Batch, Time, Channels, Height, Width]`. |
| **`audio`** | `[B, 1, 64, 300]` | `float32` | Standardized (mean~0, std~1) | 64-band Log-Mel Spectrogram (16 kHz, $n_{\text{fft}}=1024$, $\text{hop\_length}=512$, 300 time frames). **If audio stream is absent, pass a zero tensor:** `np.zeros((B, 1, 64, 300), dtype=np.float32)`. |

### Output Tensor

| Output Name | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **`logits`** | `[B, 4]` | `float32` | Raw unnormalized class scores. Apply Softmax to get probabilities. |

---

## 3. Output Classes & Anomaly Logic

| Class ID | Class Name | Is Anomaly? | Visual / Acoustic Trigger Behavior | Recommended UI / Alert Action |
| :---: | :--- | :---: | :--- | :--- |
| **`0`** | **Normal Coding** | `False` | Eyes focused on screen, natural facial movements, quiet ambient room. | Display green indicator. Normal status. |
| **`1`** | **Secondary Screen / Notes** | `True` | Persistent gaze yaw/pitch deviation (>25°) away from screen center for >5 seconds. | Display warning badge: *"Candidate Looking Away / Notes"*. |
| **`2`** | **External Voice Assistance** | `True` | Candidate mouth opening rhythmically synchronized with external voice speech formants. | Display red alert: *"External Speech / Voice Assistance Detected"*. |
| **`3`** | **Multiple Faces / Absence** | `True` | Candidate has left webcam frame (absence) OR extra person entered webcam frame. | Display urgent alert: *"Unauthorized Face Detected / Candidate Missing"*. |

---

## 4. Preprocessing Pipeline (Exact Rules)
To feed any raw OpenCV BGR frame into the model:
1. **Face Detection:** Detect candidate face using Haar cascade or MediaPipe.
2. **Crop with Margin:** Expand bounding box by 15% margin around face to retain head pose context. If no face is detected, use the center frame.
3. **Resize:** Resize crop to `(112, 112)`.
4. **Convert & Normalize:**
   ```python
   rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
   norm = (rgb.astype(np.float32) / 127.5) - 1.0
   tensor = np.transpose(norm, (2, 0, 1))  # (3, 112, 112)
   ```
5. **Sliding Temporal Window:** Buffer 16 frames. When buffer reaches 16 frames, stack into `(1, 16, 3, 112, 112)`, run ONNX inference, and slide window by 4 frames (`buffer = buffer[4:]`) for smooth real-time monitoring.

---

## 5. Ready-to-Use Python SDK Integration
A complete wrapper is implemented in `ml/src/proctor_sdk.py`.
Integration code:

```python
from ml.src.proctor_sdk import ProctoringEngine
import cv2

# Initialize engine
engine = ProctoringEngine(
    model_path="ml/models/video_proctor_model.onnx",
    window_size=16,
    slide_step=4,
    confidence_threshold=0.70
)

# Process incoming frame (from OpenCV, WebRTC, or camera stream)
# frame is standard BGR numpy array: (H, W, 3)
analysis = engine.process_frame(frame)

if analysis["status"] == "READY":
    print(f"Prediction: {analysis['label']}")
    print(f"Is Violation: {analysis['is_anomaly']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Probabilities: {analysis['probabilities']}")
```

---

## 6. HTTP / WebSocket Microservice Specification
A production FastAPI service is provided in `ml/src/api.py`.
- **Base URL:** `http://localhost:8000`
- **POST `/v1/proctor/analyze_frame`:**
  - Request body: `{"image_base64": "data:image/jpeg;base64,..."}`
  - Response JSON:
    ```json
    {
      "status": "READY",
      "is_anomaly": true,
      "class_id": 1,
      "label": "Secondary Screen / Notes",
      "confidence": 0.9852,
      "face_detected": true,
      "face_bbox": [180, 110, 240, 240],
      "latency_ms": 14.2,
      "probabilities": {
        "normal": 0.0148,
        "notes_or_secondary_screen": 0.9852,
        "voice_assistance": 0.0,
        "absence_or_multi_face": 0.0
      }
    }
    ```
- **WebSocket `/v1/proctor/stream_ws`:** Send binary JPEG frame bytes, receive real-time JSON analysis with sub-30ms response time.

---

## 7. Performance & Hardware SLA
- **Inference Latency:**
  - GPU (TensorRT / CUDA): ~15 – 25 ms per 16-frame window.
  - CPU (ONNX Runtime): ~100 – 150 ms per 16-frame window.
- **Memory Footprint:** ~250 MB RAM (ONNX Runtime session).
- **Thread Safety:** Single `InferenceSession` can be shared across async tasks or spawned per candidate session.
