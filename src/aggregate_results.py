"""Aggregate per-image ResNet JSON outputs into CSV and a summary JSON."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from statistics import mean
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def find_resnet_jsons(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("resnet_*.json"))


def extract_ground_truth(image_stem: str) -> str | None:
    """Extract ground-truth label from image stem (e.g., 'normal_000001' -> 'Normal')."""
    if image_stem.startswith("normal"):
        return "Normal"
    elif image_stem.startswith("glaucoma"):
        return "Glaucoma"
    return None


def extract_folder(json_path_str: str) -> str:
    """Extract folder name (train/val/test) from JSON path."""
    p = Path(json_path_str)
    for parent in p.parents:
        if parent.name in ("train", "val", "test"):
            return parent.name
    return "unknown"


def aggregate(results_dir: Path) -> tuple[Path, Path, Path]:
    json_paths = find_resnet_jsons(results_dir)
    if not json_paths:
        raise FileNotFoundError(f"No resnet_*.json files found in {results_dir}")

    csv_path = results_dir / "resnet_aggregate.csv"
    filtered_csv_path = results_dir / "resnet_aggregate_high_confidence.csv"
    summary_path = results_dir / "resnet_summary.json"

    rows = []
    folder_stats = defaultdict(lambda: {"correct": 0, "total": 0, "labels": defaultdict(int)})
    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for p in json_paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex:
            print(f"Skipping {p}: failed to read JSON: {ex}")
            continue

        image_stem = p.stem.replace("resnet_", "")
        ground_truth = extract_ground_truth(image_stem)
        folder = extract_folder(str(p))

        row = {
            "image_stem": image_stem,
            "json_path": str(p),
            "folder": folder,
            "label": data.get("label", ""),
            "probability": float(data.get("probability", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
            "confidence_percent": float(data.get("confidence_percent", 0.0)),
            "ground_truth": ground_truth or "unknown",
            "correct": str(ground_truth == data.get("label")) if ground_truth else "N/A",
        }
        rows.append(row)

        # Accumulate folder stats
        if ground_truth:
            folder_stats[folder]["total"] += 1
            folder_stats[folder]["labels"][ground_truth] += 1
            if ground_truth == data.get("label"):
                folder_stats[folder]["correct"] += 1

        # Accumulate confusion matrix
        if ground_truth:
            pred = data.get("label", "")
            if ground_truth == "Glaucoma" and pred == "Glaucoma":
                confusion["TP"] += 1
            elif ground_truth == "Normal" and pred == "Normal":
                confusion["TN"] += 1
            elif ground_truth == "Normal" and pred == "Glaucoma":
                confusion["FP"] += 1
            elif ground_truth == "Glaucoma" and pred == "Normal":
                confusion["FN"] += 1

    # write CSV
    fieldnames = ["image_stem", "json_path", "folder", "label", "ground_truth", "correct", "probability", "confidence", "confidence_percent"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write filtered CSV (high confidence)
    high_conf_rows = [r for r in rows if float(r["confidence_percent"]) >= 75.0]
    with open(filtered_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in high_conf_rows:
            writer.writerow(r)

    # compute summary
    total = len(rows)
    labels = {}
    probs = []
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
        probs.append(r["probability"]) if r["probability"] is not None else None

    # Compute performance metrics
    total_with_gt = sum(1 for r in rows if r["ground_truth"] != "unknown")
    correct_count = sum(1 for r in rows if r["correct"] == "True")
    accuracy = correct_count / total_with_gt if total_with_gt > 0 else 0.0

    tp, tn, fp, fn = confusion["TP"], confusion["TN"], confusion["FP"], confusion["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Folder summaries
    folder_summary = {}
    for folder, stats in sorted(folder_stats.items()):
        if stats["total"] > 0:
            folder_summary[folder] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"],
                "label_counts": dict(stats["labels"]),
            }

    summary = {
        "total_images": total,
        "label_counts": labels,
        "mean_probability": mean(probs) if probs else 0.0,
        "images_with_ground_truth": total_with_gt,
        "performance_metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "confusion_matrix": confusion,
        },
        "folder_summaries": folder_summary,
        "high_confidence_count": len(high_conf_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, summary_path, filtered_csv_path


if __name__ == "__main__":
    csv_p, summary_p, filtered_p = aggregate(RESULTS_DIR)
    print(f"Wrote aggregate CSV: {csv_p}")
    print(f"Wrote filtered CSV (confidence >= 75%): {filtered_p}")
    print(f"Wrote summary JSON: {summary_p}")
