"""
Production FastAPI Proctoring Microservice.
Exposes REST and WebSocket endpoints for real-time video stream anomaly analysis.
Run:
    uvicorn ml.src.api:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.src.proctor_sdk import ProctoringEngine

app = FastAPI(
    title="Real-Time Proctoring Anomaly Detection API",
    description="Multimodal Video Vision Transformer for interview & exam integrity monitoring",
    version="1.0.0"
)

# Enable CORS for React/Vue/Next.js frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine globally
engine = ProctoringEngine(model_path="ml/models/video_proctor_model.onnx")


class FrameAnalysisRequest(BaseModel):
    image_base64: str  # Base64-encoded JPEG/PNG


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "model": "MultimodalVideoProctorNet (36-Layer)", "runtime": "ONNX"}


@app.post("/v1/proctor/analyze_frame")
async def analyze_frame_base64(payload: FrameAnalysisRequest):
    """Analyzes a base64-encoded webcam frame from a browser."""
    # Decode base64 to OpenCV image
    img_bytes = base64.b64decode(payload.image_base64.split(",")[-1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image payload"}

    result = engine.process_frame(frame)
    return result


@app.post("/v1/proctor/upload_frame")
async def analyze_frame_upload(file: UploadFile = File(...)):
    """Analyzes an uploaded image frame file."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Could not decode uploaded image"}

    result = engine.process_frame(frame)
    return result


@app.websocket("/v1/proctor/stream_ws")
async def proctor_websocket_stream(websocket: WebSocket):
    """
    High-speed WebSocket connection for continuous live webcam streaming from candidate browser.
    Candidate client sends raw JPEG bytes; server returns instant JSON anomaly predictions.
    """
    await websocket.accept()
    session_engine = ProctoringEngine(model_path="ml/models/video_proctor_model.onnx")

    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                analysis = session_engine.process_frame(frame)
                await websocket.send_json(analysis)
    except WebSocketDisconnect:
        print("Candidate proctoring stream disconnected.")
