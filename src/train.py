import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import pandas as pd
import torch
from ultralytics import YOLO

IMGSZ = 960  # Fixed square input resolution for all training & val batches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO model with pandas CSV logging and fixed imgsz."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset/yolo_train_net-a/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base model weights (e.g., yolo11n.pt, yolo11s.pt, yolo11m.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (-1 for AutoBatch)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker threads (keep <=4 on Windows)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="../output/runs/net_defect",
        help="Project output folder",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Experiment run name",
    )
    parser.add_argument(
        "--csv-log",
        type=str,
        default=None,
        help="Custom CSV output path (defaults to <project>/<name>/custom_training_log.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        Path(args.csv_log) if args.csv_log else run_dir / "custom_training_log.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[{datetime.now().isoformat()}] Starting YOLO training")
    print(
        f"Device: {device} ({gpu_name}) | Fixed Image Size: {IMGSZ}x{IMGSZ}"
    )
    print(f"Dataset: {Path(args.data).resolve()}")
    print(f"Pandas CSV Log: {csv_path.resolve()}")

    epoch_state = {"start_time": time.time()}

    def on_train_epoch_start(trainer):
        epoch_state["start_time"] = time.time()

    def on_fit_epoch_end(trainer):
        """Logs epoch metrics to CSV via pandas immediately after each epoch."""
        elapsed = round(time.time() - epoch_state["start_time"], 2)
        metrics = trainer.metrics or {}
        loss_items = trainer.label_loss_items(trainer.tloss, prefix="train")

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": trainer.epoch + 1,
            "epochs_total": trainer.epochs,
            "imgsz": IMGSZ,
            "epoch_time_sec": elapsed,
            "train_box_loss": round(float(loss_items.get("train/box_loss", 0.0)), 6),
            "train_cls_loss": round(float(loss_items.get("train/cls_loss", 0.0)), 6),
            "train_dfl_loss": round(float(loss_items.get("train/dfl_loss", 0.0)), 6),
            "val_box_loss": round(float(metrics.get("val/box_loss", 0.0)), 6),
            "val_cls_loss": round(float(metrics.get("val/cls_loss", 0.0)), 6),
            "val_dfl_loss": round(float(metrics.get("val/dfl_loss", 0.0)), 6),
            "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 6),
            "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 6),
            "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 6),
            "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 6),
            "lr_pg0": round(float(trainer.optimizer.param_groups[0]["lr"]), 8) if trainer.optimizer else 0.0,
        }

        df_row = pd.DataFrame([row])
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        df_row.to_csv(csv_path, mode="a", header=not file_exists, index=False)

        print(
            f"Epoch {row['epoch']}/{row['epochs_total']} | "
            f"mAP50: {row['mAP50']:.4f} | "
            f"Precision: {row['precision']:.4f} | "
            f"Recall: {row['recall']:.4f} | "
            f"Time: {elapsed}s"
        )

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=IMGSZ,  
        rect=False,  # Enforce strict square batches (no variable aspect ratio)
        multi_scale=False,  # Disable random multi-scale resizing during training
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        mosaic=0.5,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        degrees=15.0,
        verbose=False,
        amp=True,  # Enable Automatic Mixed Precision (AMP) for faster training on CUDA GPUs
    )

    print(f"[{datetime.now().isoformat()}] Training finished successfully.")
    print(f"Best weights saved at: {(run_dir / 'weights' / 'best.pt').resolve()}")
    print(f"CSV metrics saved at:  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
