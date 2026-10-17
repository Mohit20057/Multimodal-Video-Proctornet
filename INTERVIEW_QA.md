# Multimodal Video Vision Transformer: Interview Q&A Playbook

This document contains the complete technical interview guide, 30-second elevator pitch, parameter cheat sheet, and in-depth architectural answers for `MultimodalVideoProctorNet`.

---

## 1. The 30-Second Elevator Pitch

> *"I designed, trained, evaluated, and exported a production-ready **36-layer, 34-Million parameter Multimodal Video Vision Transformer** named `MultimodalVideoProctorNet` for real-time behavioral anomaly detection in candidate technical interviews and proctoring environments.*
>
> *It processes synchronized **16-frame webcam face video** and **64-band Log-Mel audio spectrograms** to classify behavior into 4 categories: Normal Coding, Looking Away / Secondary Notes, External Voice Assistance, and Candidate Absence or Multiple Faces.*
>
> *I optimized the model using **PyTorch AMP FP16** with **Gradient Accumulation** on an RTX 4050 GPU, implemented **Grad-CAM spatiotemporal explainability**, and exported it to **ONNX Runtime** achieving sub-25ms GPU and sub-150ms CPU inference latency."*

---

## 2. Technical Parameter Cheat Sheet

| Parameter / Metric | Exact Specification |
| :--- | :--- |
| **Model Name** | `MultimodalVideoProctorNet` |
| **Total Trainable Parameters** | **33.9 Million** (~34M parameters) |
| **Total Hidden Layers** | **36 Layers** ($17\text{ video} + 4\text{ audio} + 2\text{ align} + 12\text{ transformer} + 1\text{ head}$) |
| **Video Input Tensor** | `[Batch, 16, 3, 112, 112]` (16 temporal frames of face-centered RGB crops, normalized to $[-1.0, 1.0]$) |
| **Audio Input Tensor** | `[Batch, 1, 64, 300]` (64-band Log-Mel Spectrogram at 16 kHz, $n_{\text{fft}}=1024$, $\text{hop\_length}=512$) |
| **Output Logits** | `[Batch, 4]` (Logits for Classes 0, 1, 2, 3) |
| **Target Classes** | **0:** Normal Coding \| **1:** Secondary Screen/Notes \| **2:** Voice Assistance \| **3:** Multi-Face/Absence |
| **Backbone Convolutions** | 17-Layer R(2+1)D ResNet (Spatial $1\times3\times3$ + Temporal $3\times1\times1$) |
| **Audio Feature Extractor** | 4-Layer 2D-CNN with BatchNorm and GELU |
| **Cross-Modal Attention** | 6 Bidirectional Blocks (12 hidden layers, 8 heads, $d_{\text{model}}=384$, $\text{FFN}=1536$) |
| **Global Pooling** | Adaptive Average Pooling $\oplus$ Adaptive Max Pooling (512-dim $\rightarrow$ 768-dim concatenated) |
| **Loss Function** | **Multiclass Focal Loss** ($\gamma = 2.0$, $\alpha = [0.25, 0.75, 0.75, 0.75]$) |
| **Optimizer & Scheduler** | `AdamW(lr=1e-4, weight_decay=1e-2)` with `CosineAnnealingWarmRestarts` |
| **Precision** | Automatic Mixed Precision (**AMP FP16**) via `torch.amp.autocast()` and `GradScaler` |
| **Effective Batch Size** | **32** (Physical batch size 16 $\times$ Gradient Accumulation Steps 2) |
| **Checkpoints & Sizes** | PyTorch Checkpoint: **407 MB** (`best_proctor_model.pt`) \| ONNX Model: **136 MB** (`video_proctor_model.onnx`) |
| **Inference Latency** | **<25 ms on GPU**, **~150 ms on CPU** per 16-frame temporal window |

---

## 3. The 4-Pillar Deep-Dive Walkthrough

### Pillar 1: Data Pipeline & Zero-Leakage Partitioning
- **Multimodal Inputs:** Dual-stream data capturing spatiotemporal facial movements (gaze drift, mouth kinetics, face presence) synchronized with acoustic audio energy.
- **Grounding Benchmarks:** Grounded on the statistical distributions of **MSU Online Exam Proctoring (OEP)**, **DAiSEE (IIT Hyderabad)**, and **Google AVA-Speech**.
- **Zero Identity Leakage:** Candidate sessions are partitioned strictly by **Session/Student ID** (70% Train, 15% Val, 15% Test) across 120 unique candidate identities. Held-out validation and test sets contain zero identities from the training set, ensuring the model generalizes to unseen human faces.

### Pillar 2: 36-Layer Architecture Breakdown
- **Why R(2+1)D Video Backbone?** Standard 3D convolutions $(3 \times 3 \times 3)$ are parameter-heavy and computationally expensive. R(2+1)D decomposes 3D filters into spatial $(1 \times 3 \times 3)$ and temporal $(3 \times 1 \times 1)$ convolutions. This introduces an additional non-linear activation between spatial and temporal operations, doubling non-linearity while reducing parameters by over 50%.
- **Why Cross-Modal Attention?** Cheating anomalies require multimodal correlation. For example, in **External Voice Assistance (Class 2)**, mouth movements must be synchronized with acoustic speech formants. Bidirectional cross-attention enables audio tokens to query facial tokens and video tokens to query audio tokens.
- **Why Dual Temporal Pooling?** Concatenating Average Pooling and Max Pooling allows the classifier to capture both sustained duration anomalies (prolonged note-reading over several seconds) and impulse peak anomalies (sudden camera walk-aways or a secondary face flashing by).

### Pillar 3: Training Recipe & Optimization
- **Multiclass Focal Loss:** Real-world proctoring telemetry is severely imbalanced (85%+ normal vs. 15% anomalies). Focal Loss modulates cross-entropy with $(1 - p_t)^\gamma$ ($\gamma=2.0$). This down-weights easy normal examples and forces gradient updates to focus on rare cheating boundaries.
- **Hardware Optimization on RTX 4050:** Using PyTorch native AMP FP16 and Gradient Accumulation (batch size 16, accumulation steps 2) saturates the GPU's Tensor Cores while capping VRAM at ~4.9 GB, avoiding CUDA out-of-memory errors on a 6 GB card.
- **Macro-Averaged F1 Checkpointing:** Checkpoint selection and early stopping monitor the unweighted Macro F1 across all 4 classes rather than raw accuracy.

### Pillar 4: Explainability & Edge Deployment
- **Grad-CAM Saliency:** Implemented 3D Grad-CAM over the final spatiotemporal convolution layer, generating visual heatmaps proving the model attends to the candidate's eyes during gaze deviations and mouth during speech anomalies.
- **ONNX Export:** Exported to `video_proctor_model.onnx` with dynamic batching. Verified numerical parity against PyTorch ($\max |y_{pt} - y_{onnx}| < 10^{-6}$) and benchmarked inference latency for real-time edge deployment.

---

## 4. Golden Answers to Follow-Up Questions

### Q1: "How do you handle class imbalance between normal candidates and cheaters?"
> *"I implemented Multiclass Focal Loss with a focusing parameter $\gamma = 2.0$ and class-weighting vector $\alpha = [0.25, 0.75, 0.75, 0.75]$. Standard cross-entropy tends to collapse to predicting Class 0 because normal behavior dominates the loss. Focal Loss exponentially down-weights well-classified examples where $p_t > 0.5$, forcing the optimizer to penalize hard anomaly false negatives 3× heavier."*

### Q2: "Why concatenate Average Pooling and Max Pooling instead of using a [CLS] token?"
> *"In continuous video action recognition, behavioral anomalies exhibit different temporal duration signatures: reading off-screen notes is a sustained event requiring high temporal mean activation, while candidate absence or camera cutoffs are sudden impulse spikes requiring peak activation. Concatentating AvgPool and MaxPool gives the dense classification head both duration context and peak intensity."*

### Q3: "What is the difference between physical batch size and gradient accumulation steps?"
> *"Physical batch size dictates how many samples reside in GPU VRAM during a single forward/backward pass. Gradient accumulation runs multiple small physical passes (e.g. 16 clips), sums their gradients without updating weights, and only calls `optimizer.step()` every $N$-th step. With `batch_size=16` and `accum_steps=2`, the model achieves the mathematical convergence of batch size 32 while requiring half the instantaneous VRAM (~4.9 GB instead of ~6.8 GB)."*

### Q4: "How does the model run in real-time production?"
> *"In production, we use the exported ONNX Runtime engine. A sliding FIFO buffer maintains the last 16 frames of candidate face crops. Every 4 frames, it feeds the tensor into ONNX Runtime, computing predictions in under 25ms on GPU (or ~150ms on CPU). If anomaly probability exceeds 0.70, a real-time integrity alert webhook is triggered."*
