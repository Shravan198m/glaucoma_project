"""Evaluation and inference utilities for the glaucoma classifier."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset import create_dataloaders
from model import create_resnet50_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"


def build_inference_transform() -> transforms.Compose:
    """Match the validation/test preprocessing used during training."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_trained_model(model_path: str | os.PathLike[str] | None = None) -> torch.nn.Module:
    """Load the trained ResNet-50 checkpoint for inference."""
    checkpoint_path = Path(model_path) if model_path is not None else MODEL_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained model not found: {checkpoint_path}")

    model = create_resnet50_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_single_image(
    image_path: str | os.PathLike[str],
    model_path: str | os.PathLike[str] | None = None,
) -> Dict[str, object]:
    """Predict glaucoma probability for a single fundus image."""
    model = load_trained_model(model_path)
    transform = build_inference_transform()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        probability = float(model(input_tensor).item())

    label = "Glaucoma" if probability >= 0.5 else "Normal"
    confidence = probability if label == "Glaucoma" else 1.0 - probability

    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 2),
    }


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return TN, FP, FN, TP for binary labels."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    total = tn + fp + fn + tp

    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute a simple ROC curve and AUC without third-party dependencies."""
    thresholds = np.r_[np.inf, np.sort(np.unique(y_prob))[::-1], -np.inf]
    fpr_values = []
    tpr_values = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    fpr_array = np.array(fpr_values, dtype=float)
    tpr_array = np.array(tpr_values, dtype=float)
    auc = float(np.trapz(tpr_array, fpr_array))
    return fpr_array, tpr_array, auc


def evaluate_model(
    model_path: str | os.PathLike[str] | None = None,
    dataset_root: str | os.PathLike[str] | None = None,
    batch_size: int = 8,
) -> Dict[str, object]:
    """Evaluate the saved model on the held-out test set and save plots."""
    checkpoint_path = Path(model_path) if model_path is not None else MODEL_PATH
    root = Path(dataset_root) if dataset_root is not None else PROJECT_ROOT / "dataset"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained model not found: {checkpoint_path}")

    _, _, test_loader, _ = create_dataloaders(str(root), batch_size=batch_size)
    model = load_trained_model(checkpoint_path)

    y_true: list[int] = []
    y_prob: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            y_prob.extend(outputs.squeeze(1).tolist())
            y_true.extend(labels.tolist())

    y_true_array = np.array(y_true, dtype=int)
    y_prob_array = np.array(y_prob, dtype=float)
    y_pred_array = (y_prob_array >= 0.5).astype(int)

    metrics = _classification_metrics(y_true_array, y_pred_array)
    fpr, tpr, auc = _roc_curve(y_true_array, y_prob_array)
    metrics["auc"] = auc

    plots_root = PROJECT_ROOT / "outputs" / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    confusion_path = plots_root / "confusion_matrix.png"
    roc_path = plots_root / "roc_curve.png"

    _save_confusion_matrix(metrics, confusion_path)
    _save_roc_curve(fpr, tpr, auc, roc_path)

    return {
        "metrics": metrics,
        "confusion_matrix_path": str(confusion_path),
        "roc_curve_path": str(roc_path),
    }


def _save_confusion_matrix(metrics: Dict[str, float], save_path: Path) -> None:
    """Save a confusion matrix heatmap."""
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Normal", "Glaucoma"])
    ax.set_yticks([0, 1], labels=["Normal", "Glaucoma"])

    for row in range(2):
        for col in range(2):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: Path) -> None:
    """Save the ROC curve plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the evaluation pipeline on the saved test split."""
    result = evaluate_model()
    metrics = result["metrics"]

    print("Evaluation complete.")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Confusion matrix saved to: {result['confusion_matrix_path']}")
    print(f"ROC curve saved to: {result['roc_curve_path']}")


if __name__ == "__main__":
    main()
