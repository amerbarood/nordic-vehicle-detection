"""Downloading the YOLOv9 code, pretrained weights and the NVD dataset, and preparing data.yaml."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

YOLOV9_DIR = Path("yolov9-main")
YOLOV9_ZIP_URL = "https://github.com/WongKinYiu/yolov9/archive/refs/heads/main.zip"
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt"
WEIGHTS_PATH = Path("data/models/yolov9-e.pt")
NVD_URL = (
    "https://filedn.eu/lwnF6OGJjyNyTUCGJUbsnVp/public_data/project_nvd/"
    "Labeled%20Frames%20(YOLO%20Format).zip"
)
DATASET_DIR = Path("data/nvd")
SPLITS_DIR = Path("data/splits")
CONFIGS_DIR = Path("configs")
LOCAL_DATA_YAML = CONFIGS_DIR / "data.local.yaml"


def _download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}")

    def progress(blocks, block_size, total):
        if total > 0:
            print(f"  {min(blocks * block_size / total, 1):.0%}", end="\r")

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    tmp.rename(dest)
    print()
    return dest


def download_yolov9() -> Path:
    """Download and unpack the official YOLOv9 repository into yolov9-main/."""
    if (YOLOV9_DIR / "train_dual.py").exists():
        print(f"YOLOv9 code already in {YOLOV9_DIR}/")
        return YOLOV9_DIR
    archive = _download(YOLOV9_ZIP_URL, Path("data/downloads/yolov9-main.zip"))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(".")
    print(f"Unpacked to {YOLOV9_DIR}/. Install its requirements with: pip install -r {YOLOV9_DIR}/requirements.txt")
    return YOLOV9_DIR


def download_weights() -> Path:
    return _download(WEIGHTS_URL, WEIGHTS_PATH)


def find_dataset_root(base: Path) -> Path:
    """Locate the folder that contains images/ and labels/ inside an unpacked archive."""
    base = Path(base)
    if (base / "images").is_dir():
        return base
    for candidate in sorted(base.rglob("images")):
        if candidate.is_dir() and (candidate.parent / "labels").is_dir():
            return candidate.parent
    raise FileNotFoundError(f"No folder with images/ and labels/ found under {base}")


def download_dataset() -> Path:
    """Download (about 24 GB) and unpack the NVD 'Labeled Frames (YOLO format)' archive."""
    archive = _download(NVD_URL, Path("data/downloads/nvd_labeled_frames_yolo.zip"))
    if not any(DATASET_DIR.glob("**/images")):
        print(f"Unpacking to {DATASET_DIR}/ ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATASET_DIR)
    return find_dataset_root(DATASET_DIR)


def prepare_dataset(root: Path | None = None) -> Path:
    """Copy the train/val/test split lists next to the images and write configs/data.local.yaml.

    The split lists reference images as ./images/<name>.png relative to the dataset root.
    """
    root = find_dataset_root(root or DATASET_DIR).resolve()
    missing = 0
    for split in ("train", "val", "test"):
        src = SPLITS_DIR / f"{split}.txt"
        shutil.copy(src, root / f"{split}.txt")
        missing += sum(1 for line in src.read_text().splitlines() if line.strip() and not (root / line.strip()).exists())
    if missing:
        print(f"Warning: {missing} images listed in the splits were not found under {root}")
    CONFIGS_DIR.mkdir(exist_ok=True)
    LOCAL_DATA_YAML.write_text(
        f"path: {root.as_posix()}  # dataset root dir\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n\n"
        "nc: 1\n"
        "names: ['Car']\n"
    )
    print(f"Wrote {LOCAL_DATA_YAML} (dataset root: {root})")
    return LOCAL_DATA_YAML
