"""Final prediction pipeline for glaucoma detection.

Runs preprocessing, optic-disc ROI extraction, segmentation, CDR computation,
and ResNet-50 inference for one image (or a folder of images).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cdr import compute_cdr, interpret_cdr
from model import create_resnet50_model
from preprocessing import find_first_image, preprocess_image
from segmentation import detect_optic_disc_roi, segment_disc_and_cup


# Same normalization as training.
predict_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_single_image(
    image_path: str | Path, model_path: str | Path = "outputs/models/best_model.pth"
) -> Dict[str, Any]:
    """Run the full glaucoma prediction pipeline for one image."""
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("\n" + "=" * 60)
    print("GLAUCOMA DETECTION SYSTEM - PREDICTION")
    print("=" * 60)
    print(f"Image: {image_path.name}")

    device = torch.device("cpu")

    # Step 1: Load model.
    model = create_resnet50_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Step 2: Preprocessing.
    print("\n[1/5] Preprocessing image...")
    prep_results = preprocess_image(str(image_path), use_clahe=True)

    # Step 3: ROI Extraction.
    print("[2/5] Locating optic disc...")
    roi, center, bbox = detect_optic_disc_roi(prep_results["normalized"])

    # Step 4: Segmentation.
    print("[3/5] Segmenting disc and cup...")
    disc_mask, cup_mask, _, _ = segment_disc_and_cup(roi)

    # Step 5: CDR Calculation.
    print("[4/5] Computing CDR...")
    cdr, cdr_details = compute_cdr(disc_mask, cup_mask)
    cdr_interpretation = interpret_cdr(cdr)
    print(f"      CDR = {cdr:.4f} -> {cdr_interpretation['status']}")

    # Step 6: CNN Classification.
    print("[5/5] Running ResNet-50 classification...")
    image_pil = Image.open(image_path).convert("RGB")
    input_tensor = predict_transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = float(output.item())

    cnn_prediction = "GLAUCOMA" if probability >= 0.5 else "NORMAL"
    confidence = probability if probability >= 0.5 else (1 - probability)
    confidence_pct = confidence * 100.0

    print(f"      CNN: {cnn_prediction} (confidence: {confidence_pct:.1f}%)")

    # Final decision (combining CDR + CNN).
    final_diagnosis = combine_decisions(cdr, cnn_prediction)

    # Display/save results.
    display_final_output(
        image_path=image_path,
        prep_results=prep_results,
        roi=roi,
        disc_mask=disc_mask,
        cup_mask=cup_mask,
        cdr=cdr,
        cdr_details=cdr_details,
        cnn_pred=cnn_prediction,
        confidence=confidence_pct,
        final_diagnosis=final_diagnosis,
        center=center,
        bbox=bbox,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  CDR Value:       {cdr:.4f}")
    print(f"  CNN Prediction:  {cnn_prediction} ({confidence_pct:.1f}%)")
    print(f"  Final Diagnosis: {final_diagnosis['label']}")
    print(f"  Recommendation:  {final_diagnosis['recommendation']}")
    print("=" * 60)

    return {
        "image": image_path.name,
        "cdr": cdr,
        "cnn_prediction": cnn_prediction,
        "cnn_confidence": confidence_pct,
        "final_diagnosis": final_diagnosis["label"],
        "recommendation": final_diagnosis["recommendation"],
        "optic_disc_center": center,
        "optic_disc_bbox": bbox,
    }


def combine_decisions(cdr: float, cnn_prediction: str) -> Dict[str, str]:
    """Combine CDR and CNN outputs into one final diagnosis.

    Decision logic:
    - If both CDR and CNN agree: high-confidence result
    - If they disagree: manual review recommended
    """
    cdr_positive = cdr >= 0.6

    if cdr_positive and cnn_prediction == "GLAUCOMA":
        return {
            "label": "GLAUCOMA - HIGH RISK",
            "confidence": "HIGH",
            "recommendation": "Immediate specialist consultation required.",
            "color": "red",
        }
    if (not cdr_positive) and cnn_prediction == "NORMAL":
        return {
            "label": "NORMAL",
            "confidence": "HIGH",
            "recommendation": "Routine ophthalmological checkup (annual).",
            "color": "green",
        }

    return {
        "label": "BORDERLINE - MANUAL REVIEW NEEDED",
        "confidence": "LOW",
        "recommendation": (
            f"CDR suggests {'Glaucoma' if cdr_positive else 'Normal'} "
            f"but CNN predicts {cnn_prediction}. "
            f"Please consult an ophthalmologist."
        ),
        "color": "orange",
    }


def display_final_output(
    image_path: Path,
    prep_results: Dict[str, np.ndarray],
    roi: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    cdr_details: Dict[str, Any],
    cnn_pred: str,
    confidence: float,
    final_diagnosis: Dict[str, str],
    center: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> None:
    """Display and save a complete visual diagnostic output."""
    del cdr_details, center, bbox  # Reserved for future report expansion.

    def to_green(gray: np.ndarray) -> np.ndarray:
        """Convert a grayscale or float image to green-tinted RGB for display."""
        if gray.dtype != np.uint8:
            img = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            img = gray.copy()
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 1] = img
        return rgb

    original = prep_results["original_rgb"]
    green_channel = prep_results["green_channel"]
    clahe_image = prep_results["clahe_enhanced"]
    normalized_image = prep_results["normalized"]
    cup_mask_fixed = (cup_mask.astype(np.uint8) * disc_mask.astype(np.uint8)).astype(np.uint8)

    disc_rows = np.where(disc_mask > 0)[0]
    cup_rows = np.where(cup_mask_fixed > 0)[0]
    disc_vertical = int(disc_rows.max() - disc_rows.min() + 1) if disc_rows.size else 1
    cup_vertical = int(cup_rows.max() - cup_rows.min() + 1) if cup_rows.size else 0
    cdr = cup_vertical / disc_vertical if disc_vertical > 0 else 0.0

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    fig.patch.set_facecolor("#1a1a2e")

    # Row 1: preprocessing pipeline.
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original RGB", color="white", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(to_green(green_channel))
    axes[0, 1].set_title("Green channel", color="white", fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(to_green(clahe_image))
    axes[0, 2].set_title("CLAHE enhanced", color="white", fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(to_green(normalized_image))
    axes[0, 3].set_title("Normalized [0, 1]", color="white", fontweight="bold")
    axes[0, 3].axis("off")

    # Row 2: ROI + masks.
    axes[1, 0].imshow(to_green(roi))
    axes[1, 0].set_title("Optic disc ROI", color="white", fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(disc_mask * 255, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Optic disc mask", color="white", fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cup_mask_fixed * 255, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Optic cup mask", color="white", fontweight="bold")
    axes[1, 2].axis("off")

    seg_overlay = np.zeros((*roi.shape, 3), dtype=np.float32)
    seg_overlay[:, :, 1] = roi
    disc_coords = np.where(disc_mask > 0)
    cup_coords = np.where(cup_mask_fixed > 0)
    seg_overlay[disc_coords[0], disc_coords[1]] = [0.0, 0.9, 0.0]
    seg_overlay[cup_coords[0], cup_coords[1]] = [1.0, 1.0, 0.0]
    axes[1, 3].imshow(np.clip(seg_overlay, 0, 1))
    disc_patch = mpatches.Patch(color="green", label="Optic disc")
    cup_patch = mpatches.Patch(color="yellow", label="Optic cup")
    axes[1, 3].legend(
        handles=[disc_patch, cup_patch],
        loc="lower right",
        fontsize=8,
        facecolor="black",
        labelcolor="white",
    )
    axes[1, 3].set_title("Segmentation overlay", color="white", fontweight="bold")
    axes[1, 3].axis("off")

    # Row 3: results.
    ax5 = axes[2, 0]
    cdr_color = "red" if cdr >= 0.6 else "green"
    ax5.bar(["CDR"], [cdr], color=cdr_color, width=0.4, edgecolor="white")
    ax5.axhline(y=0.6, color="yellow", linestyle="--", linewidth=2, label="Threshold (0.6)")
    ax5.set_ylim(0, 1.0)
    ax5.set_title("Cup-to-Disc Ratio", color="white", fontweight="bold")
    ax5.set_facecolor("#16213e")
    ax5.tick_params(colors="white")
    ax5.spines["bottom"].set_color("white")
    ax5.spines["left"].set_color("white")
    ax5.text(0, cdr + 0.02, f"{cdr:.3f}", ha="center", color="white", fontweight="bold", fontsize=14)
    ax5.legend(facecolor="#16213e", labelcolor="white")

    ax6 = axes[2, 1]
    probs = [100.0 - confidence if cnn_pred == "GLAUCOMA" else confidence, confidence if cnn_pred == "GLAUCOMA" else 100.0 - confidence]
    ax6.bar(["Normal", "Glaucoma"], probs, color=["green", "red"], edgecolor="white")
    ax6.set_ylim(0, 100)
    ax6.set_title("CNN Prediction Confidence", color="white", fontweight="bold")
    ax6.set_facecolor("#16213e")
    ax6.tick_params(colors="white")
    ax6.spines["bottom"].set_color("white")
    ax6.spines["left"].set_color("white")
    ax6.set_ylabel("Confidence (%)", color="white")

    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    result_color = final_diagnosis["color"]
    result_text = (
        "DIAGNOSIS REPORT\n"
        + ("-" * 30)
        + "\n\n"
        + f"CDR Value:  {cdr:.3f} (after fix)\n"
        + "Threshold:  0.6\n\n"
        + f"CNN Result: {cnn_pred}\n"
        + f"Confidence: {confidence:.1f}%\n\n"
        + ("-" * 30)
        + "\n\n"
        + f"FINAL:\n{final_diagnosis['label']}\n\n"
        + f"Recommendation:\n{final_diagnosis['recommendation']}"
    )

    axes[2, 2].text(
        0.05,
        0.95,
        result_text,
        transform=axes[2, 2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        color="white",
        bbox=dict(boxstyle="round", facecolor="#0f3460", edgecolor=result_color, linewidth=3),
    )

    plt.suptitle(
        "Glaucoma Detection System - Complete Diagnostic Output",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.01,
    )

    plt.tight_layout()
    os.makedirs("outputs/results", exist_ok=True)
    save_name = f"outputs/results/result_{image_path.stem}.png"
    plt.savefig(save_name, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")

    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    print(f"\nResult saved to {save_name}")


def predict_folder(
    folder_path: str | Path, model_path: str | Path = "outputs/models/best_model.pth"
) -> List[Dict[str, Any]]:
    """Run predictions on all image files in a folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in sorted(folder_path.iterdir()) if p.is_file() and p.suffix.lower() in extensions]

    print(f"\nFound {len(images)} images in {folder_path}")
    results: List[Dict[str, Any]] = []

    for i, image_file in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {image_file.name}")
        result = predict_single_image(image_file, model_path)
        results.append(result)

    if results:
        print("\n" + "=" * 70)
        print("BATCH PREDICTION SUMMARY")
        print("=" * 70)
        print(f"{'Image':<30} {'CDR':>6} {'CNN':>10} {'Confidence':>12} {'Final'}")
        print("-" * 70)

        for r in results:
            print(
                f"{r['image']:<30} "
                f"{r['cdr']:>6.3f} "
                f"{r['cnn_prediction']:>10} "
                f"{r['cnn_confidence']:>11.1f}% "
                f"{r['final_diagnosis']}"
            )

    return results


if __name__ == "__main__":
    # Example:
    # predict_folder("dataset/test/glaucoma")
    # predict_folder("dataset/test/normal")
    default_image = Path("dataset/test/glaucoma/sample.jpg")
    if default_image.exists():
        predict_single_image(default_image)
    else:
        fallback = find_first_image(Path("dataset/test/glaucoma"))
        if fallback is None:
            raise FileNotFoundError(
                "No test image found in dataset/test/glaucoma."
            )
        predict_single_image(fallback)
