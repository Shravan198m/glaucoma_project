"""CDR (Cup-to-Disc Ratio) calculation for glaucoma assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_vertical_diameter(mask: np.ndarray) -> int:
    """Compute the vertical diameter of a binary mask.

    WHY VERTICAL CDR?
    - Clinicians measure CDR vertically (top to bottom of disc/cup).
    - Vertical measurement is more sensitive to glaucomatous changes.
    - It's the clinical standard (not area-based).

    Method:
    1. For each column in the mask, find topmost and bottommost white pixel.
    2. The vertical extent of each column gives a height measurement.
    3. Average across all columns with disc/cup pixels.

    Alternative: Find the overall bounding box height.
    We use bounding box height (simpler and robust for binary masks).
    """
    if np.sum(mask) == 0:
        return 0  # Empty mask

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return 0

    row_min = coords[0].min()
    row_max = coords[0].max()

    vertical_diameter = row_max - row_min + 1  # In pixels
    return int(vertical_diameter)


def compute_horizontal_diameter(mask: np.ndarray) -> int:
    """Compute the horizontal diameter of a binary mask.

    Included for completeness (some papers use horizontal CDR).
    """
    if np.sum(mask) == 0:
        return 0

    coords = np.where(mask > 0)
    col_min = coords[1].min()
    col_max = coords[1].max()

    return int(col_max - col_min + 1)


def compute_cdr(
    disc_mask: np.ndarray, cup_mask: np.ndarray
) -> Tuple[float, Dict[str, float | int]]:
    """Compute Cup-to-Disc Ratio (CDR).

    Formula: CDR = Vertical Cup Diameter / Vertical Disc Diameter

    Clinical thresholds:
    - CDR < 0.5: Normal (low risk)
    - CDR 0.5-0.6: Borderline (monitor closely)
    - CDR > 0.6: Glaucoma suspected (refer to specialist)
    - CDR > 0.8: High suspicion of advanced glaucoma

    Returns:
        cdr: float — the computed CDR value
        details: dict — breakdown of measurements
    """
    disc_vertical = compute_vertical_diameter(disc_mask)
    cup_vertical = compute_vertical_diameter(cup_mask)

    disc_horizontal = compute_horizontal_diameter(disc_mask)
    cup_horizontal = compute_horizontal_diameter(cup_mask)

    disc_area = int(np.sum(disc_mask))
    cup_area = int(np.sum(cup_mask))

    # Vertical CDR (primary metric)
    if disc_vertical == 0:
        print("⚠️  Warning: Disc vertical diameter is 0. CDR cannot be computed.")
        cdr = 0.0
    else:
        cdr = float(cup_vertical) / float(disc_vertical)

    # Area-based CDR (secondary metric, for comparison)
    area_cdr = np.sqrt(float(cup_area) / float(disc_area)) if disc_area > 0 else 0.0

    details = {
        "disc_vertical_diameter": disc_vertical,
        "cup_vertical_diameter": cup_vertical,
        "disc_horizontal_diameter": disc_horizontal,
        "cup_horizontal_diameter": cup_horizontal,
        "disc_area_pixels": disc_area,
        "cup_area_pixels": cup_area,
        "vertical_cdr": round(cdr, 4),
        "area_cdr": round(float(area_cdr), 4),
    }

    return cdr, details


def interpret_cdr(cdr: float) -> Dict[str, str]:
    """Provide clinical interpretation of CDR value."""
    if cdr < 0.5:
        status = "NORMAL"
        risk = "Low Risk"
        recommendation = "Routine ophthalmological checkup (annual)"
        color = "green"
    elif cdr < 0.6:
        status = "BORDERLINE"
        risk = "Moderate Risk"
        recommendation = "Monitor closely. Follow-up in 6 months."
        color = "orange"
    elif cdr < 0.8:
        status = "GLAUCOMA SUSPECTED"
        risk = "High Risk"
        recommendation = "Immediate specialist consultation required."
        color = "red"
    else:
        status = "ADVANCED GLAUCOMA SUSPECTED"
        risk = "Very High Risk"
        recommendation = "Urgent specialist consultation. Further tests required."
        color = "darkred"

    return {
        "status": status,
        "risk_level": risk,
        "recommendation": recommendation,
        "color": color,
    }


def display_cdr_report(
    roi_image: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    details: Dict[str, float | int],
    save_path: str | None = None,
) -> Dict[str, str]:
    """Display a complete CDR report with visualization and optional saving."""
    interpretation = interpret_cdr(cdr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Build overlay image
    overlay = np.stack([roi_image] * 3, axis=-1)

    disc_coords = np.where(disc_mask > 0)
    cup_coords = np.where(cup_mask > 0)

    # Disc in green
    overlay[disc_coords[0], disc_coords[1], 0] = 0.0
    overlay[disc_coords[0], disc_coords[1], 1] = 0.8
    overlay[disc_coords[0], disc_coords[1], 2] = 0.0

    # Cup in yellow
    overlay[cup_coords[0], cup_coords[1], 0] = 1.0
    overlay[cup_coords[0], cup_coords[1], 1] = 1.0
    overlay[cup_coords[0], cup_coords[1], 2] = 0.0

    axes[0].imshow(np.clip(overlay, 0, 1))
    axes[0].set_title("Segmentation Overlay\n(Green=Disc, Yellow=Cup)", fontweight="bold")
    axes[0].axis("off")

    # CDR bar chart
    categories = ["Disc Diameter\n(vertical)", "Cup Diameter\n(vertical)"]
    values = [details["disc_vertical_diameter"], details["cup_vertical_diameter"]]
    colors = ["#2ecc71", "#f39c12"]

    bars = axes[1].bar(categories, values, color=colors, width=0.5, edgecolor="black")
    axes[1].set_ylabel("Pixels", fontsize=11)
    axes[1].set_title(f'CDR Measurement\nVertical CDR = {cdr:.3f}', fontweight="bold")

    for bar, val in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            fontweight="bold",
        )

    # CDR gauge / report
    axes[2].axis("off")
    report_text = (
        f"CDR REPORT\n"
        f"{'─'*35}\n\n"
        f"Vertical CDR:     {cdr:.3f}\n"
        f"Area CDR:         {details['area_cdr']:.3f}\n\n"
        f"Disc Diameter:    {details['disc_vertical_diameter']} px\n"
        f"Cup Diameter:     {details['cup_vertical_diameter']} px\n\n"
        f"Disc Area:        {details['disc_area_pixels']} px²\n"
        f"Cup Area:         {details['cup_area_pixels']} px²\n\n"
        f"{'─'*35}\n\n"
        f"Status: {interpretation['status']}\n\n"
        f"Risk: {interpretation['risk_level']}\n\n"
        f"Recommendation:\n{interpretation['recommendation']}"
    )

    color = interpretation["color"]
    axes[2].text(
        0.05,
        0.95,
        report_text,
        transform=axes[2].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor=color, linewidth=2),
    )

    plt.suptitle("Cup-to-Disc Ratio Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=150, bbox_inches="tight")
        print(f"Saved CDR report to {save_path_obj}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

    return interpretation


def main() -> None:
    """Run the full preprocessing -> ROI -> segmentation -> CDR demo."""
    from pathlib import Path

    from preprocessing import find_first_image

    project_root = Path(__file__).resolve().parents[1]
    sample_folder = project_root / "dataset" / "train" / "normal"
    sample_image = find_first_image(sample_folder)
    output_plot = project_root / "outputs" / "plots" / f"cdr_{sample_image.stem if sample_image else 'demo'}.png"

    if sample_image is None:
        raise FileNotFoundError(
            f"No sample image found in {sample_folder}. Add dataset images and try again."
        )

    print(f"Sample image: {sample_image}")

    pipeline = run_full_pipeline(str(sample_image), save_outputs=True)
    details = pipeline["details"]
    cdr = pipeline["cdr"]
    interpretation = pipeline["interpretation"]

    print("\n" + "=" * 40)
    print("CDR RESULTS:")
    print("=" * 40)
    for key, val in details.items():
        print(f"  {key}: {val}")
    print(f"\nFinal CDR: {cdr:.4f}")
    print(f"\nDiagnosis: {interpretation['status']}")
    print(f"Recommendation: {interpretation['recommendation']}")


def run_full_pipeline(image_path: str, save_outputs: bool = True) -> Dict[str, object]:
    """Run preprocessing, ROI extraction, segmentation, and CDR on one image.

    This keeps the preprocessed green-channel-derived image as the shared input
    for the rest of the pipeline so the stages flow in sequence.
    """
    from preprocessing import preprocess_image
    from segmentation import detect_optic_disc_roi, segment_disc_and_cup

    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "outputs" / "plots"

    print("\nStage 1 - Preprocessing")
    results = preprocess_image(image_path)
    normalized_image = results["normalized"]
    print(f"  Normalized image shape: {normalized_image.shape}")

    print("\nStage 2 - ROI Extraction")
    roi, center, bbox = detect_optic_disc_roi(normalized_image)
    print(f"  ROI shape: {roi.shape}")
    print(f"  ROI center: {center}")
    print(f"  ROI bbox: {bbox}")

    print("\nStage 3 - K-Strange Segmentation")
    disc_mask, cup_mask, _, _ = segment_disc_and_cup(roi)

    print("\nStage 4 - CDR Calculation")
    cdr, details = compute_cdr(disc_mask, cup_mask)
    interpretation = interpret_cdr(cdr)

    artifacts: Dict[str, object] = {
        "image_path": image_path,
        "preprocessing": results,
        "normalized_image": normalized_image,
        "roi": roi,
        "roi_center": center,
        "roi_bbox": bbox,
        "disc_mask": disc_mask,
        "cup_mask": cup_mask,
        "cdr": cdr,
        "details": details,
        "interpretation": interpretation,
    }

    if save_outputs:
        outputs_root.mkdir(parents=True, exist_ok=True)
        artifact_name = Path(image_path).stem
        preprocess_plot = outputs_root / f"preprocessing_{artifact_name}.png"
        segmentation_plot = outputs_root / f"segmentation_{artifact_name}.png"
        cdr_plot = outputs_root / f"cdr_{artifact_name}.png"

        from preprocessing import visualize_preprocessing
        from segmentation import visualize_segmentation

        visualize_preprocessing(results, save_path=str(preprocess_plot))
        visualize_segmentation(roi, disc_mask, cup_mask, save_path=segmentation_plot)
        display_cdr_report(roi, disc_mask, cup_mask, cdr, details, save_path=str(cdr_plot))

        artifacts["saved_plots"] = {
            "preprocessing": str(preprocess_plot),
            "segmentation": str(segmentation_plot),
            "cdr": str(cdr_plot),
        }

    return artifacts


if __name__ == "__main__":
    main()
