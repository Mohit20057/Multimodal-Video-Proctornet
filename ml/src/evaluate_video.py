"""
Comprehensive Evaluation, Explainability (Grad-CAM), and ONNX Export Suite
for MultimodalVideoProctorNet.
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve, auc
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from video_model import get_model
from dataset_loader import get_dataloaders


# ==============================================================================
# 1. Grad-CAM Implementation for Spatiotemporal Video Conv Feature Maps
# ==============================================================================

class VideoGradCAM:
    """
    Grad-CAM for MultimodalVideoProctorNet's spatiotemporal video backbone.
    Highlights eye, mouth, and head-pose facial regions responsible for anomaly flags.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Target the final spatiotemporal convolution layer
        target_layer = self.model.video_backbone.stage4[-1].conv2.temporal_conv

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self, video: torch.Tensor, audio: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap for a single sample (1, 16, 3, 112, 112).
        Returns: (16, 112, 112) spatial attention map across all 16 frames.
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(video, audio)  # (1, 4)
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Activations: (1, 512, 16, H', W')
        # Gradients: (1, 512, 16, H', W')
        grads = self.gradients.detach()
        acts = self.activations.detach()

        # Global average pool gradients over spatial & temporal dimensions: (1, 512, 1, 1, 1)
        weights = torch.mean(grads, dim=(2, 3, 4), keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, 16, H', W')
        cam = F.relu(cam)

        # Upsample to full video resolution (16, 112, 112)
        cam = F.interpolate(cam, size=(16, 112, 112), mode='trilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


# ==============================================================================
# 2. Evaluation & Metric Reporting
# ==============================================================================

def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str = "ml/models"
):
    """
    Evaluates test performance, saves normalized confusion matrix,
    and computes per-class Precision, Recall, F1, and PR-AUC.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    print("\n--- Running Test Set Evaluation ---")
    with torch.no_grad():
        for videos, audios, targets in test_loader:
            videos = videos.to(device)
            audios = audios.to(device)

            logits = model(videos, audios)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    class_names = [
        "0: Normal Coding",
        "1: Second Monitor / Notes",
        "2: Voice Assistance",
        "3: Multi-Face / Absence"
    ]

    # Classification Report
    print("\nDetailed Per-Class Performance:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))

    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f"Overall Test Macro F1-Score: {macro_f1:.4f}")

    # Compute PR-AUC per class
    print("\nPrecision-Recall AUC (PR-AUC) per Class:")
    for c in range(4):
        bin_targets = (all_targets == c).astype(int)
        if np.sum(bin_targets) > 0:
            precision, recall, _ = precision_recall_curve(bin_targets, all_probs[:, c])
            pr_auc = auc(recall, precision)
            print(f"  • {class_names[c]}: PR-AUC = {pr_auc:.4f}")

    # Normalized Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix (MultimodalVideoProctorNet)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Label values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2%}",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground Truth Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")


# ==============================================================================
# 3. ONNX Production Export & Parity Benchmark
# ==============================================================================

def export_and_benchmark_onnx(
    model: nn.Module,
    onnx_path: str = "ml/models/video_proctor_model.onnx",
    device: torch.device = torch.device('cpu')
):
    """
    Exports model to ONNX with dynamic batching, verifies parity with PyTorch,
    and benchmarks inference latency.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ONNX or onnxruntime not found. Skipping ONNX export.")
        return

    print("\n--- Exporting to ONNX Runtime ---")
    model.eval().to(device)

    dummy_video = torch.randn(1, 16, 3, 112, 112, device=device)
    dummy_audio = torch.randn(1, 1, 64, 300, device=device)

    # PyTorch reference forward pass
    with torch.no_grad():
        pt_out = model(dummy_video, dummy_audio).cpu().numpy()

    torch.onnx.export(
        model,
        (dummy_video, dummy_audio),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['video', 'audio'],
        output_names=['logits'],
        dynamic_axes={
            'video': {0: 'batch_size'},
            'audio': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"ONNX model successfully exported to: {onnx_path}")

    # Numeric parity verification
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        'video': dummy_video.cpu().numpy(),
        'audio': dummy_audio.cpu().numpy()
    }
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Numeric Parity Check (Max Absolute Difference): {max_diff:.6e}")
    assert max_diff < 1e-4, f"Parity check failed! Max diff: {max_diff}"
    print("[PASS] Parity verification PASSED (max_diff < 1e-4)")

    # Inference Latency Benchmark
    print("\n--- Benchmarking ONNX Inference Latency ---")
    num_runs = 50
    # Warmup
    for _ in range(5):
        _ = ort_session.run(None, ort_inputs)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"Average Latency: {mean_lat:.2f} ms per 16-frame window")
    print(f"P95 Latency:     {p95_lat:.2f} ms per 16-frame window")
    if mean_lat < 65.0:
        print("[PASS] Latency SLA achieved (<65 ms CPU inference budget)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultimodalVideoProctorNet")
    parser.add_argument("--checkpoint", type=str, default="ml/models/best_proctor_model.pt")
    parser.add_argument("--dataset", type=str, default="ml/data/proctoring_dataset.h5")
    parser.add_argument("--output_dir", type=str, default="ml/models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=4).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"Checkpoint {args.checkpoint} not found. Running with initialized weights.")

    # Load test dataloader
    _, _, test_loader = get_dataloaders(hdf5_path=args.dataset, batch_size=16)

    # 1. Evaluate Metrics & Confusion Matrix
    evaluate_test_set(model, test_loader, device=device, output_dir=args.output_dir)

    # 2. Grad-CAM Visualization
    print("\n--- Generating Grad-CAM Saliency Explanation ---")
    gradcam = VideoGradCAM(model)
    sample_v, sample_a, sample_y = test_loader.dataset[0]
    sample_v = sample_v.unsqueeze(0).to(device)
    sample_a = sample_a.unsqueeze(0).to(device)

    cam_map = gradcam.generate_heatmap(sample_v, sample_a, target_class=sample_y.item())
    gradcam.remove_hooks()

    plt.figure(figsize=(10, 4))
    plt.imshow(cam_map[8], cmap='jet')
    plt.colorbar()
    plt.title(f"Grad-CAM Temporal Mid-Frame (Class {sample_y.item()} Facial Trigger)")
    gradcam_path = os.path.join(args.output_dir, "gradcam_explanation.png")
    plt.savefig(gradcam_path, dpi=200)
    plt.close()
    print(f"Grad-CAM explanation saved to: {gradcam_path}")

    # 3. ONNX Export & Parity Benchmark
    onnx_path = os.path.join(args.output_dir, "video_proctor_model.onnx")
    export_and_benchmark_onnx(model, onnx_path=onnx_path, device=torch.device('cpu'))


if __name__ == "__main__":
    main()
