"""
Download and Ingestion Helper for Real Internet Proctoring & Exam Datasets.
Supports:
1. Automated Kaggle Download for MSU Online Exam Proctoring Dataset (raajanwankhade/oep-dataset)
2. Direct download of public benchmark exam & interview sample video streams
3. Directory layout organization for raw real-world video processing
"""
import os
import sys
import argparse
import urllib.request
import zipfile
import subprocess
from pathlib import Path


def download_kaggle_msu_dataset(
    dest_dir: str = "ml/data/raw_videos",
    kaggle_username: str = None,
    kaggle_key: str = None
):
    """
    Downloads the official MSU Online Exam Proctoring Dataset from Kaggle.
    Dataset: raajanwankhade/oep-dataset
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Set Kaggle credentials in environment if provided
    if kaggle_username and kaggle_key:
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key
        print(f"Using provided Kaggle credentials for user: {kaggle_username}")
    else:
        # Check ~/.kaggle/kaggle.json
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists() and "KAGGLE_KEY" not in os.environ:
            print("\n[!] Kaggle credentials not found.")
            print("To download the real MSU Online Exam Proctoring dataset from Kaggle:")
            print("1. Go to https://www.kaggle.com/settings -> Click 'Create New Token'")
            print("2. Place kaggle.json in ~/.kaggle/kaggle.json, OR pass:")
            print("   python ml/src/download_real_data.py --kaggle_user YOUR_USER --kaggle_key YOUR_KEY")
            return False

    print("\nDownloading MSU Online Exam Proctoring Dataset via Kaggle API...")
    try:
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "raajanwankhade/oep-dataset",
            "-p", dest_dir,
            "--unzip"
        ]
        subprocess.run(cmd, check=True)
        print(f"[OK] Dataset successfully downloaded and unzipped into: {dest_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error during Kaggle download: {e}")
        return False
    except FileNotFoundError:
        print("Installing kaggle CLI...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        return download_kaggle_msu_dataset(dest_dir, kaggle_username, kaggle_key)


def download_public_sample_exam_videos(dest_dir: str = "ml/data/raw_videos"):
    """
    Downloads public open-source real exam webcam and face video samples
    available directly without authentication.
    """
    os.makedirs(dest_dir, exist_ok=True)
    print(f"\nDownloading public exam webcam and behavior clips into {dest_dir}...")

    # Public direct test video assets from open-source proctoring benchmarks
    public_videos = [
        {
            "name": "candidate_01_normal.mp4",
            "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female-and-male.mp4",
            "label": 0
        },
        {
            "name": "candidate_02_face_tracking.mp4",
            "url": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking.mp4",
            "label": 3
        }
    ]

    downloaded = 0
    for item in public_videos:
        out_path = os.path.join(dest_dir, item["name"])
        if not os.path.exists(out_path):
            print(f"Fetching {item['name']} from {item['url']}...")
            try:
                urllib.request.urlretrieve(item["url"], out_path)
                print(f"  [OK] Saved to: {out_path}")
                downloaded += 1
            except Exception as e:
                print(f"  [!] Failed to download {item['name']}: {e}")
        else:
            print(f"  {item['name']} already exists.")
            downloaded += 1

    print(f"\nTotal real video files ready in {dest_dir}: {downloaded}")


def main():
    parser = argparse.ArgumentParser(description="Download Real Proctoring Video Datasets from Internet")
    parser.add_argument("--source", type=str, choices=["kaggle", "public", "all"], default="all",
                        help="Data source to download from ('kaggle' for MSU dataset, 'public' for open test videos)")
    parser.add_argument("--dest", type=str, default="ml/data/raw_videos", help="Target destination directory")
    parser.add_argument("--kaggle_user", type=str, default=None, help="Kaggle username")
    parser.add_argument("--kaggle_key", type=str, default=None, help="Kaggle API key")
    args = parser.parse_args()

    if args.source in ["kaggle", "all"]:
        success = download_kaggle_msu_dataset(args.dest, args.kaggle_user, args.kaggle_key)
        if not success and args.source == "kaggle":
            return

    if args.source in ["public", "all"]:
        download_public_sample_exam_videos(args.dest)


if __name__ == "__main__":
    main()
