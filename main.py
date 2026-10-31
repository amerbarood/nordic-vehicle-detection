"""Command-line entry point.

    python main.py setup --code --weights          # YOLOv9 repo + yolov9-e.pt
    python main.py setup --dataset                 # NVD frames (~24 GB) + configs/data.local.yaml
    python main.py setup --prepare /path/to/nvd    # already have the dataset: just write data.local.yaml
    python main.py train                           # 50-epoch fine-tuning (CUDA GPU)
    python main.py validate runs/train/nvd_yolov9e/weights/best.pt --task test
    python main.py detect runs/train/nvd_yolov9e/weights/best.pt --sample 5
    python main.py baseline --sample 5             # pretrained YOLOv9-E without fine-tuning
    python main.py results                         # summarise / plot results/results.csv

Add --dry-run to train/validate/detect to print the YOLOv9 command without running it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvd import results, setup_env, yolo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv9 vehicle detection on Nordic drone footage")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="download code, weights and data")
    p.add_argument("--code", action="store_true", help="download the official YOLOv9 repository")
    p.add_argument("--weights", action="store_true", help="download COCO-pretrained yolov9-e.pt")
    p.add_argument("--dataset", action="store_true", help="download and unpack the NVD labeled frames (~24 GB)")
    p.add_argument("--prepare", type=Path, metavar="DATASET_ROOT", nargs="?", const=setup_env.DATASET_DIR,
                   help="write configs/data.local.yaml for a dataset folder with images/ and labels/")

    t = yolo.TrainArgs()
    p = sub.add_parser("train", help="fine-tune YOLOv9-E with train_dual.py")
    p.add_argument("--epochs", type=int, default=t.epochs)
    p.add_argument("--batch", type=int, default=t.batch)
    p.add_argument("--img", type=int, default=t.img)
    p.add_argument("--workers", type=int, default=t.workers)
    p.add_argument("--device", default=t.device)
    p.add_argument("--data", type=Path, default=t.data)
    p.add_argument("--weights", type=Path, default=t.weights)
    p.add_argument("--name", default=t.name)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("validate", help="compute precision/recall/mAP with val_dual.py")
    p.add_argument("weights", type=Path)
    p.add_argument("--task", choices=["val", "test"], default="val")
    p.add_argument("--data", type=Path, default=setup_env.LOCAL_DATA_YAML)
    p.add_argument("--img", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("detect", help="run detection with detect.py (1280 px, conf 0.1 by default)")
    p.add_argument("weights", type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="image, folder, video or glob")
    src.add_argument("--sample", type=int, help="number of random test-split images")
    p.add_argument("--img", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("baseline", help="pretrained YOLOv9-E (Ultralytics) without fine-tuning")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, nargs="+")
    src.add_argument("--sample", type=int)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--car-only", action="store_true", help="keep only COCO 'car' detections")

    p = sub.add_parser("results", help="summarise and plot a training run's results.csv")
    p.add_argument("csv", type=Path, nargs="?", default=results.DEFAULT_RESULTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "setup":
        if not any([args.code, args.weights, args.dataset, args.prepare]):
            raise SystemExit("Choose at least one of --code, --weights, --dataset, --prepare")
        if args.code:
            setup_env.download_yolov9()
        if args.weights:
            setup_env.download_weights()
        if args.dataset:
            setup_env.prepare_dataset(setup_env.download_dataset())
        elif args.prepare:
            setup_env.prepare_dataset(args.prepare)

    elif args.command == "train":
        a = yolo.TrainArgs(epochs=args.epochs, batch=args.batch, img=args.img, workers=args.workers,
                           device=args.device, data=args.data, weights=args.weights, name=args.name)
        raise SystemExit(yolo.run(yolo.train_command(a), args.dry_run))

    elif args.command == "validate":
        cmd = yolo.val_command(args.weights, args.data, args.img, device=args.device, task=args.task)
        raise SystemExit(yolo.run(cmd, args.dry_run))

    elif args.command == "detect":
        sources = [args.source] if args.source else yolo.sample_images(args.sample, seed=args.seed)
        code = 0
        for source in sources:
            code |= yolo.run(yolo.detect_command(args.weights, source, args.img, args.conf, args.device), args.dry_run)
        print(f"Detections are saved under {yolo.RUNS_DIR / 'detect'}/")
        raise SystemExit(code)

    elif args.command == "baseline":
        from nvd import baseline

        images = args.images or yolo.sample_images(args.sample)
        baseline.run(images, args.conf, args.car_only)

    elif args.command == "results":
        df = results.load_results(args.csv)
        print(results.summary(df).round(3).to_string())
        results.plot(df)


if __name__ == "__main__":
    main()
