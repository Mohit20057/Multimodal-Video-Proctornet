"""
All-in-One End-to-End Runner for MultimodalVideoProctorNet.
Executes:
1. Data Preparation (if not already built)
2. 36-Layer Video Transformer Training with AMP on GPU
3. Test Set Evaluation, Confusion Matrix, Grad-CAM Saliency & ONNX Export
"""
import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="End-to-End Proctoring Pipeline Runner")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of benchmark samples to generate if missing")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    args = parser.parse_args()

    python_bin = sys.executable
    dataset_path = "ml/data/proctoring_dataset.h5"

    print("=" * 60)
    print("🚀 STARTING MULTIMODAL VIDEO PROCTORING END-TO-END PIPELINE")
    print("=" * 60)

    # Step 1: Check or build dataset
    if not os.path.exists(dataset_path):
        print("\n[Step 1/3] Dataset not found. Generating benchmark proctoring telemetry...")
        cmd_prep = [
            python_bin, "ml/src/video_preprocessor.py",
            "--mode", "synthetic",
            "--num_samples", str(args.num_samples),
            "--output", dataset_path
        ]
        subprocess.run(cmd_prep, check=True)
    else:
        print(f"\n[Step 1/3] Existing dataset found at: {dataset_path} (Skipping creation)")

    # Step 2: Training
    print("\n[Step 2/3] Starting 36-Layer Neural Network Training on GPU...")
    cmd_train = [
        python_bin, "ml/src/train_video.py",
        "--dataset", dataset_path,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--accum_steps", str(args.accum_steps),
        "--save_path", "ml/models/best_proctor_model.pt"
    ]
    subprocess.run(cmd_train, check=True)

    # Step 3: Evaluation, Grad-CAM & ONNX Export
    print("\n[Step 3/3] Evaluating Test Set, Generating Grad-CAM & Exporting ONNX...")
    cmd_eval = [
        python_bin, "ml/src/evaluate_video.py",
        "--checkpoint", "ml/models/best_proctor_model.pt",
        "--dataset", dataset_path,
        "--output_dir", "ml/models"
    ]
    subprocess.run(cmd_eval, check=True)

    print("\n" + "=" * 60)
    print("✅ ENTIRE PIPELINE COMPLETED SUCCESSFULLY!")
    print("Outputs available at:")
    print("  • Best PyTorch Model: ml/models/best_proctor_model.pt")
    print("  • Production ONNX:    ml/models/video_proctor_model.onnx")
    print("  • Confusion Matrix:   ml/models/confusion_matrix.png")
    print("  • Grad-CAM Heatmap:   ml/models/gradcam_explanation.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
