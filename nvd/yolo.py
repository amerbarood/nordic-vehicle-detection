"""Wrappers around the official YOLOv9 scripts (train_dual.py, val_dual.py, detect.py)."""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .setup_env import LOCAL_DATA_YAML, WEIGHTS_PATH, YOLOV9_DIR, find_dataset_root, DATASET_DIR

CUSTOM_CFG = Path("configs/yolov9_custom.yaml")
HYP = Path("configs/hyp.scratch-high.yaml")
RUNS_DIR = Path("runs")


@dataclass
class TrainArgs:
    """Defaults are the final 50-epoch run reported in the README."""

    epochs: int = 50
    batch: int = 6
    img: int = 640
    workers: int = 12
    device: str = "0"
    data: Path = LOCAL_DATA_YAML
    weights: Path = WEIGHTS_PATH
    cfg: Path = CUSTOM_CFG
    hyp: Path = HYP
    name: str = "nvd_yolov9e"


def _script(name: str) -> Path:
    path = YOLOV9_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py setup --code` first.")
    return path


def train_command(a: TrainArgs) -> list[str]:
    return [
        sys.executable, str(_script("train_dual.py")),
        "--workers", str(a.workers), "--batch-size", str(a.batch), "--img", str(a.img), "--epochs", str(a.epochs),
        "--data", str(a.data), "--weights", str(a.weights), "--cfg", str(a.cfg), "--hyp", str(a.hyp),
        "--device", a.device, "--project", str((RUNS_DIR / "train").resolve()), "--name", a.name,
    ]


def val_command(weights: Path, data: Path = LOCAL_DATA_YAML, img: int = 640, batch: int = 6, device: str = "0",
                task: str = "val") -> list[str]:
    return [
        sys.executable, str(_script("val_dual.py")),
        "--weights", str(weights), "--data", str(data), "--img", str(img), "--batch-size", str(batch),
        "--device", device, "--task", task, "--project", str((RUNS_DIR / "val").resolve()), "--name", task,
    ]


def detect_command(weights: Path, source: Path | str, img: int = 1280, conf: float = 0.1, device: str = "0") -> list[str]:
    return [
        sys.executable, str(_script("detect.py")),
        "--weights", str(weights), "--source", str(source), "--img", str(img), "--conf-thres", str(conf),
        "--device", device, "--project", str((RUNS_DIR / "detect").resolve()), "--name", "exp",
    ]


def run(cmd: list[str], dry_run: bool = False) -> int:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry_run:
        return 0
    # YOLOv9 scripts resolve their own modules relative to the yolov9-main folder.
    return subprocess.run(cmd, check=False).returncode


def sample_images(n: int, split: str = "test", seed: int | None = None, root: Path | None = None) -> list[Path]:
    """Pick n random images from a split list (data/splits/<split>.txt)."""
    root = find_dataset_root(root or DATASET_DIR)
    names = [line.strip() for line in Path(f"data/splits/{split}.txt").read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    return [root / name for name in rng.sample(names, min(n, len(names)))]
