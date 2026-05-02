"""Unified glaucoma pipeline runner for the end-to-end demo flow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw, ImageFont
import json

from cdr import run_full_pipeline
from evaluate import predict_single_image
from preprocessing import find_first_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "results"
MODELS_ROOT = PROJECT_ROOT / "outputs" / "models"


def _combine_decisions(cdr_value: float, cnn_label: str, cnn_confidence: float) -> tuple[str, str, float]:
    """Combine the rule-based CDR result with the CNN prediction."""
    cdr_label = "Glaucoma" if cdr_value >= 0.6 else "Normal"
    cdr_confidence = min(1.0, 0.5 + abs(cdr_value - 0.6) * 1.25)

    if cdr_label == cnn_label:
        final_label = cdr_label
        final_confidence = min(1.0, (cdr_confidence + cnn_confidence) / 2.0 + 0.1)
        note = "CDR and CNN agree."
    else:
        final_label = "Borderline"
        final_confidence = max(0.5, min(cdr_confidence, cnn_confidence))
        note = "CDR and CNN disagree. Manual review needed."

    return final_label, note, final_confidence


def _save_final_report(
    image_path: Path,
    pipeline_result: dict,
    cnn_result: dict | None,
    final_label: str,
    note: str,
    final_confidence: float,
) -> Path:
    """Save a combined visual summary of the end-to-end result."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_ROOT / f"final_result_{image_path.stem}.png"

    original = np.array(Image.open(image_path).convert("RGB"))
    roi = pipeline_result["roi"]
    disc_mask = pipeline_result["disc_mask"]
    cup_mask = pipeline_result["cup_mask"]
    cdr_value = float(pipeline_result["cdr"])
    interpretation = pipeline_result["interpretation"]

    overlay = np.stack([roi] * 3, axis=-1).astype(np.float32) / 255.0 if roi.dtype != np.float32 else np.stack([roi] * 3, axis=-1)
    disc_coords = np.where(disc_mask > 0)
    cup_coords = np.where(cup_mask > 0)
    overlay[disc_coords[0], disc_coords[1], 0] = 0.0
    overlay[disc_coords[0], disc_coords[1], 1] = 0.8
    overlay[disc_coords[0], disc_coords[1], 2] = 0.0
    overlay[cup_coords[0], cup_coords[1], 0] = 1.0
    overlay[cup_coords[0], cup_coords[1], 1] = 1.0
    overlay[cup_coords[0], cup_coords[1], 2] = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original)
    axes[0].set_title("Input Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("ROI + Segmentation", fontweight="bold")
    axes[1].axis("off")

    axes[2].axis("off")
    cnn_block = (
        "CNN RESULT\n"
        f"Label: {cnn_result['label'] if cnn_result else 'Unavailable'}\n"
        f"Confidence: {cnn_result['confidence_percent'] if cnn_result else 0.0:.2f}%\n\n"
        "CDR RESULT\n"
        f"CDR: {cdr_value:.3f}\n"
        f"Status: {interpretation['status']}\n\n"
        "FINAL RESULT\n"
        f"Label: {final_label}\n"
        f"Confidence: {final_confidence * 100.0:.2f}%\n"
        f"Note: {note}"
    )
    axes[2].text(
        0.03,
        0.97,
        cnn_block,
        transform=axes[2].transAxes,
        fontsize=11,
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black", linewidth=1.5),
    )

    fig.suptitle("Glaucoma Pipeline Result", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved final report to {output_path}")
    return output_path


def main() -> None:
    """Run the unified demo pipeline on one fundus image."""
    image_folder = PROJECT_ROOT / "dataset" / "train" / "normal"
    sample_image = find_first_image(image_folder)
    if sample_image is None:
        raise FileNotFoundError(f"No sample image found in {image_folder}")

    print("Unified glaucoma pipeline")
    print(f"Sample image: {sample_image}")
    print("Stages 1-4: preprocessing -> ROI -> segmentation -> CDR")
    pipeline_result = run_full_pipeline(str(sample_image), save_outputs=True)

    cnn_result = None
    model_path = MODELS_ROOT / "best_model.pth"
    if model_path.exists():
        print("Stage 5-7: CNN inference and final decision")
        cnn_result = predict_single_image(sample_image, model_path)
        print(f"CNN label: {cnn_result['label']}")
        print(f"CNN confidence: {cnn_result['confidence_percent']:.2f}%")
        # persist per-image CNN outputs
        _save_cnn_outputs(sample_image, cnn_result)
    else:
        print(f"CNN model not found at {model_path}; skipping CNN inference.")

    cdr_value = float(pipeline_result["cdr"])
    cdr_label = "Glaucoma" if cdr_value >= 0.6 else "Normal"
    cdr_confidence = min(1.0, 0.5 + abs(cdr_value - 0.6) * 1.25)

    if cnn_result is not None:
        final_label, note, final_confidence = _combine_decisions(
            cdr_value, cnn_result["label"], float(cnn_result["confidence"])
        )
    else:
        final_label = cdr_label
        note = "Only CDR-based decision available."
        final_confidence = cdr_confidence

    output_path = _save_final_report(
        sample_image,
        pipeline_result,
        cnn_result,
        final_label,
        note,
        final_confidence,
    )

    print(f"Final label: {final_label}")
    print(f"Final confidence: {final_confidence * 100.0:.2f}%")
    print(f"Final note: {note}")
    print(f"Result image: {output_path}")


def _save_cnn_outputs(image_path: Path, cnn_result: dict) -> tuple[Path, Path]:
    """Save a JSON with CNN outputs and an annotated PNG next to results."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_ROOT / f"resnet_{image_path.stem}.json"
    annotated_path = RESULTS_ROOT / f"resnet_{image_path.stem}.png"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(cnn_result, fh, indent=2)
    print(f"Saved JSON result to: {json_path}")

    # Annotate and save image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    text_lines = [f"Label: {cnn_result['label']}", f"Prob: {cnn_result['probability']:.6f}", f"Confidence: {cnn_result['confidence_percent']}%"]
    x, y = 10, 10
    for line in text_lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 20

    img.save(annotated_path)
    print(f"Saved annotated image to: {annotated_path}")
    return json_path, annotated_path


def batch_inference(dataset_root: Path | None = None) -> None:
    """Run CNN inference over all images in dataset and save per-image outputs.

    Scans `train`, `val`, `test` subfolders recursively and writes one JSON+PNG per image.
    """
    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "dataset"

    model_path = MODELS_ROOT / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    image_paths = list(dataset_root.rglob("*.png")) + list(dataset_root.rglob("*.jpg")) + list(dataset_root.rglob("*.jpeg"))
    image_paths = [p for p in image_paths if p.is_file()]
    print(f"Found {len(image_paths)} images under {dataset_root}")

    for idx, img_path in enumerate(sorted(image_paths)):
        print(f"[{idx+1}/{len(image_paths)}] Inferring: {img_path}")
        try:
            res = predict_single_image(img_path, model_path)
            _save_cnn_outputs(img_path, res)
        except Exception as ex:
            print(f"Failed on {img_path}: {ex}")


if __name__ == "__main__":
    # Support optional "batch" CLI argument
    if len(sys.argv) > 1 and sys.argv[1].lower() == "batch":
        batch_inference()
    else:
        main()
