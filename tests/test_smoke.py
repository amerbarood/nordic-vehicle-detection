"""Checks that need no GPU, dataset or YOLOv9 download."""

from pathlib import Path

import pytest

from nvd import results, setup_env, yolo


def test_results_summary_matches_readme():
    df = results.load_results()
    s = results.summary(df)
    assert list(s.index) == [10, 20, 30, 50]
    assert s.loc[50, "metrics/mAP_0.5"] == pytest.approx(0.973, abs=1e-3)
    assert s.loc[50, "metrics/mAP_0.5:0.95"] == pytest.approx(0.757, abs=1e-3)


def test_prepare_dataset(tmp_path, monkeypatch):
    root = tmp_path / "unzipped" / "Labeled Frames"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    monkeypatch.setattr(setup_env, "LOCAL_DATA_YAML", tmp_path / "data.local.yaml")
    monkeypatch.setattr(setup_env, "CONFIGS_DIR", tmp_path)
    out = setup_env.prepare_dataset(tmp_path / "unzipped")
    text = Path(out).read_text()
    assert f"path: {root.resolve().as_posix()}" in text and "names: ['Car']" in text
    assert (root / "train.txt").exists()


def test_train_command(monkeypatch, tmp_path):
    fake = tmp_path / "yolov9-main"
    fake.mkdir()
    (fake / "train_dual.py").touch()
    monkeypatch.setattr(yolo, "YOLOV9_DIR", fake)
    cmd = yolo.train_command(yolo.TrainArgs())
    joined = " ".join(cmd)
    assert "--epochs 50" in joined and "--batch-size 6" in joined and "hyp.scratch-high.yaml" in joined


def test_missing_yolov9_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(yolo, "YOLOV9_DIR", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="setup --code"):
        yolo.detect_command(Path("best.pt"), "img.png")
