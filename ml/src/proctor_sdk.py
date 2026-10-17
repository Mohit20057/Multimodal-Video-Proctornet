"""
Plug-and-Play Proctoring SDK for easy integration into existing Python projects.
Usage:
    from proctor_sdk import ProctoringEngine

    engine = ProctoringEngine()
    result = engine.process_frame(opencv_frame)
    if result["is_anomaly"]:
        print(f"Alert: {result['label']} (Confidence: {result['confidence']:.1%})")
"""
import os
import time
import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, Any, Optional


class ProctoringEngine:
    """
    Lightweight, thread-safe inference wrapper around video_proctor_model.onnx.
    Maintains a sliding temporal window and outputs structured JSON alerts.
    """
    CLASS_NAMES = [
        "Normal Coding",
        "Secondary Screen / Notes",
        "External Voice Assistance",
        "Multiple Faces / Absence"
    ]

    def __init__(
        self,
        model_path: str = "ml/models/video_proctor_model.onnx",
        window_size: int = 16,
        slide_step: int = 4,
        confidence_threshold: float = 0.70
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # Initialize ONNX Runtime with CPU or CUDA if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.window_size = window_size
        self.slide_step = slide_step
        self.confidence_threshold = confidence_threshold

        # OpenCV Haar cascade for face centering
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_buffer = []

    def process_frame(self, frame_bgr: np.ndarray, audio_spectrogram: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Processes a single incoming video frame (e.g. from OpenCV or WebRTC).
        Returns an anomaly analysis dictionary.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        # 1. Face localization and cropping
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            margin = int(0.15 * max(fw, fh))
            x1, y1 = max(0, fx - margin), max(0, fy - margin)
            x2, y2 = min(w, fx + fw + margin), min(h, fy + fh + margin)
            crop = frame_bgr[y1:y2, x1:x2]
            face_detected = True
            face_bbox = [int(fx), int(fy), int(fw), int(fh)]
        else:
            crop = frame_bgr
            face_detected = False
            face_bbox = None

        # 2. Resize to 112x112 and normalize to [-1.0, 1.0]
        crop_rgb = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2RGB)
        crop_norm = (crop_rgb.astype(np.float32) / 127.5) - 1.0
        crop_norm = np.transpose(crop_norm, (2, 0, 1))  # (3, 112, 112)

        self.frame_buffer.append(crop_norm)

        # 3. If window is not full yet, return buffering status
        if len(self.frame_buffer) < self.window_size:
            return {
                "status": "BUFFERING",
                "buffer_fill": len(self.frame_buffer) / self.window_size,
                "is_anomaly": False,
                "label": "Initializing...",
                "confidence": 0.0,
                "face_detected": face_detected,
                "face_bbox": face_bbox
            }

        # 4. Prepare batch for ONNX runtime
        video_tensor = np.expand_dims(np.array(self.frame_buffer, dtype=np.float32), axis=0)  # (1, 16, 3, 112, 112)

        if audio_spectrogram is None:
            audio_spectrogram = np.zeros((1, 1, 64, 300), dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {'video': video_tensor, 'audio': audio_spectrogram})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = outputs[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        is_anomaly = (pred_class != 0) and (confidence >= self.confidence_threshold)

        # Slide window
        self.frame_buffer = self.frame_buffer[self.slide_step:]

        return {
            "status": "READY",
            "is_anomaly": is_anomaly,
            "class_id": pred_class,
            "label": self.CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4),
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "latency_ms": round(latency_ms, 2),
            "probabilities": {
                "normal": round(float(probs[0]), 4),
                "notes_or_secondary_screen": round(float(probs[1]), 4),
                "voice_assistance": round(float(probs[2]), 4),
                "absence_or_multi_face": round(float(probs[3]), 4)
            }
        }
