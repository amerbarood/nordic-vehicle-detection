"""Summaries and plots of a YOLOv9 training run's results.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_RESULTS = Path("results/results.csv")


def load_results(path: Path = DEFAULT_RESULTS) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def summary(df: pd.DataFrame, epochs=(10, 20, 30)) -> pd.DataFrame:
    """mAP at selected epochs (1-based, as in the README) plus the final epoch."""
    final = int(df["epoch"].max()) + 1
    wanted = sorted({e for e in epochs if e <= final} | {final})
    rows = df.set_index(df["epoch"] + 1).loc[wanted]
    return rows[["metrics/precision", "metrics/recall", "metrics/mAP_0.5", "metrics/mAP_0.5:0.95"]].rename_axis("epoch")


def plot(df: pd.DataFrame, out: Path = Path("outputs/training_metrics.png")) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epoch = df["epoch"] + 1
    for col in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
        axes[0].plot(epoch, df[col], label=col.split("/")[1])
    axes[0].set_yscale("log")
    axes[0].set_title("Training losses")
    for col in ("metrics/precision", "metrics/recall"):
        axes[1].plot(epoch, df[col], label=col.split("/")[1])
    axes[1].set_title("Validation precision / recall")
    for col in ("metrics/mAP_0.5", "metrics/mAP_0.5:0.95"):
        axes[2].plot(epoch, df[col], label=col.split("/")[1])
    axes[2].set_title("Validation mAP")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out
