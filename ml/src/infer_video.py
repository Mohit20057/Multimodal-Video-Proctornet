"""
Real-Time Inference and Live Proctoring HUD using Exported ONNX Model.
Can run on:
1. Video files: python ml/src/infer_video.py --video ml/data/raw_videos/candidate_01_normal.mp4
2. Live Webcam: python ml/src/infer_video.py --webcam
"""
import os
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

CLASS_NAMES = [
    "Normal Coding",
    "Secondary Screen / Notes",
    "External Voice Assistance",
    "Multiple Faces / Absence"
]

CLASS_COLORS = [
    (0, 255, 0),    # Green: Normal
    (0, 165, 255),  # Orange: Secondary Screen
    (0, 0, 255),    # Red: Voice Assistance
    (0, 0, 255)     # Red: Absence / Multi-face
]


def run_inference(
    video_source,
    onnx_path="ml/models/video_proctor_model.onnx",
    window_frames=16,
    target_size=(112, 112)
):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    print(f"Loading ONNX runtime engine: {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Open video source (0 for webcam or file path)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video stream: {video_source}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_buffer = []
    fps_time = time.time()
    curr_label = 0
    curr_conf = 1.0

    print("\nStarting Proctoring Vision Engine... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        # Crop face if found, otherwise center-crop
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fx, fy, fw, fh = faces[0]
            margin = int(0.15 * max(fw, fh))
            x1, y1 = max(0, fx - margin), max(0, fy - margin)
            x2, y2 = min(w, fx + fw + margin), min(h, fy + fh + margin)
            face_crop = frame[y1:y2, x1:x2]
            # Draw bounding box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), CLASS_COLORS[curr_label], 2)
            cv2.putText(frame, f"Faces Detected: {len(faces)}", (fx, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[curr_label], 2)
        else:
            # Absence or no clear face detected
            face_crop = frame

        # Resize to 112x112 and normalize
        crop_rgb = cv2.cvtColor(cv2.resize(face_crop, target_size), cv2.COLOR_BGR2RGB)
        crop_norm = (crop_rgb.astype(np.float32) / 127.5) - 1.0  # (112, 112, 3)
        crop_norm = np.transpose(crop_norm, (2, 0, 1))          # (3, 112, 112)

        frame_buffer.append(crop_norm)

        # Once we have 16 temporal frames, run ONNX inference!
        if len(frame_buffer) == window_frames:
            # Video tensor: (1, 16, 3, 112, 112)
            video_input = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)

            # Dummy ambient audio spectrogram: (1, 1, 64, 300)
            audio_input = np.zeros((1, 1, 64, 300), dtype=np.float32)

            t0 = time.perf_counter()
            outputs = session.run(None, {'video': video_input, 'audio': audio_input})
            lat = (time.perf_counter() - t0) * 1000.0

            logits = outputs[0][0]
            # Softmax
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)

            curr_label = int(np.argmax(probs))
            curr_conf = float(probs[curr_label])

            # Slide window by 4 frames for smooth continuous inference
            frame_buffer = frame_buffer[4:]

        # Draw Heads-Up Display (HUD)
        status_text = f"{CLASS_NAMES[curr_label]} ({curr_conf:.1%})"
        badge_color = CLASS_COLORS[curr_label]

        # Header background
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(frame, "PROCTORING AI ENGINE", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"STATUS: {status_text}", (15, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2)

        # FPS indicator
        fps = 1.0 / max(1e-5, time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Real-Time Behavioral Proctoring HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Proctoring Inference HUD")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4/.avi)")
    parser.add_argument("--webcam", action="store_true", help="Use live laptop webcam")
    parser.add_argument("--onnx", type=str, default="ml/models/video_proctor_model.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    src = 0 if args.webcam or args.video is None else args.video
    run_inference(video_source=src, onnx_path=args.onnx)
