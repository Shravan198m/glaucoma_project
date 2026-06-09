"""Segmentation utilities for optic disc and cup regions."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")


def find_kmin_kmax(intensity_values: np.ndarray) -> Tuple[float, float]:
    """Return the minimum and maximum intensities used as K-Strange anchors."""
    kmin_val = float(np.min(intensity_values))
    kmax_val = float(np.max(intensity_values))

    print(f"  Kmin intensity: {kmin_val:.4f} (darkest pixel)")
    print(f"  Kmax intensity: {kmax_val:.4f} (brightest pixel)")
    print(f"  Intensity range: {kmax_val - kmin_val:.4f}")

    return kmin_val, kmax_val


def assign_clusters(
    intensity_values: np.ndarray, kmin_val: float, kmax_val: float
) -> np.ndarray:
    """Assign each pixel to the nearest anchor in intensity space."""
    dist_to_kmin = np.abs(intensity_values - kmin_val)
    dist_to_kmax = np.abs(intensity_values - kmax_val)

    labels = np.where(dist_to_kmin <= dist_to_kmax, 0, 1)

    cluster0_count = int(np.sum(labels == 0))
    cluster1_count = int(np.sum(labels == 1))
    print(f"  Cluster 0 (dark/background): {cluster0_count} pixels")
    print(f"  Cluster 1 (bright/disc): {cluster1_count} pixels")

    return labels


def kstrange_segment(
    roi_image: np.ndarray, stage_name: str = "Stage"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one round of K-Strange clustering (K=2) to an image patch."""
    print(f"\n--- K-Strange Clustering: {stage_name} ---")

    h, w = roi_image.shape
    intensities = roi_image.flatten()

    kmin_val, kmax_val = find_kmin_kmax(intensities)
    labels_1d = assign_clusters(intensities, kmin_val, kmax_val)
    labels_2d = labels_1d.reshape(h, w)

    cluster0_mask = (labels_2d == 0).astype(np.uint8)
    cluster1_mask = (labels_2d == 1).astype(np.uint8)

    return cluster0_mask, cluster1_mask, labels_2d


def refine_disc_mask_with_ellipse(disc_mask: np.ndarray) -> np.ndarray:
    """Refine the disc mask by fitting and filling an ellipse around it."""
    contours, _ = cv2.findContours(
        disc_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return disc_mask.astype(np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 5:
        return disc_mask.astype(np.uint8)

    ellipse = cv2.fitEllipse(largest_contour)
    refined_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    cv2.ellipse(refined_mask, ellipse, 1, thickness=-1)

    return refined_mask


def segment_disc_and_cup(
    roi_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage K-Strange segmentation to extract optic disc and optic cup."""
    print("\n" + "=" * 50)
    print("ENHANCED K-STRANGE SEGMENTATION")
    print("=" * 50)

    print("\n[STAGE 1] Separating Optic Disc from Background...")
    bg_mask, disc_mask_raw, stage1_labels = kstrange_segment(
        roi_image, "Disc vs Background"
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disc_mask_clean = cv2.morphologyEx(disc_mask_raw, cv2.MORPH_CLOSE, kernel)
    disc_mask_clean = cv2.morphologyEx(disc_mask_clean, cv2.MORPH_OPEN, kernel)
    disc_mask_refined = refine_disc_mask_with_ellipse(disc_mask_clean)
    print("  Refined disc mask with fitted ellipse")

    print("\n[STAGE 2] Separating Optic Cup from Disc Tissue...")
    disc_region = roi_image.copy()
    disc_region[disc_mask_refined == 0] = 0

    disc_pixel_count = int(np.sum(disc_mask_refined))
    print(f"  Disc pixels available for Stage 2: {disc_pixel_count}")

    if disc_pixel_count < 100:
        print("  Too few disc pixels. Cup mask set to empty.")
        cup_mask = np.zeros_like(disc_mask_refined)
        stage2_labels = np.zeros_like(disc_mask_refined)
    else:
        disc_intensities = disc_region[disc_mask_refined == 1]
        disc_y, disc_x = np.where(disc_mask_refined > 0)
        disc_center_x = float(np.mean(disc_x))
        disc_center_y = float(np.mean(disc_y))

        kmin_val = float(np.min(disc_intensities))
        kmax_val = float(np.max(disc_intensities))
        cup_threshold = float(np.percentile(disc_intensities, 65.0))
        print(f"  Disc region Kmin: {kmin_val:.4f}, Kmax: {kmax_val:.4f}")
        print(f"  Cup threshold (65th percentile): {cup_threshold:.4f}")
        print(f"  Disc centroid: ({disc_center_x:.1f}, {disc_center_y:.1f})")

        cup_mask_raw = ((disc_region >= cup_threshold) & (disc_mask_refined == 1)).astype(np.uint8)

        cup_mask = cv2.morphologyEx(cup_mask_raw, cv2.MORPH_CLOSE, kernel)
        cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel)
        cup_mask = (cup_mask.astype(np.uint8) * disc_mask_refined.astype(np.uint8)).astype(np.uint8)

        # Keep the component closest to the disc center so the cup stays central.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cup_mask, connectivity=8)
        if num_labels > 1:
            disc_area = int(np.sum(disc_mask_refined))
            best_label = 0
            best_score = -1.0

            for label in range(1, num_labels):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < max(10, int(disc_area * 0.02)):
                    continue

                component_mask = labels == label
                component_y, component_x = np.where(component_mask)
                if component_x.size == 0:
                    continue

                centroid_x = float(np.mean(component_x))
                centroid_y = float(np.mean(component_y))
                center_distance = float(np.hypot(centroid_x - disc_center_x, centroid_y - disc_center_y))
                compactness = area / max(1, stats[label, cv2.CC_STAT_WIDTH] * stats[label, cv2.CC_STAT_HEIGHT])
                score = (area * compactness) / (1.0 + center_distance)

                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label == 0:
                best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

            cup_mask = (labels == best_label).astype(np.uint8)

        stage2_labels = cup_mask

    print("\nSegmentation complete!")
    print(f"  Disc pixels: {int(np.sum(disc_mask_refined))}")
    print(f"  Cup pixels: {int(np.sum(cup_mask))}")

    return disc_mask_refined, cup_mask, stage1_labels, stage2_labels


def visualize_segmentation(
    roi_image: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    save_path: Path | str | None = None,
) -> None:
    """Visualize the segmentation results and optionally save the figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(roi_image, cmap="gray")
    axes[0].set_title("Original ROI\n(Optic Disc Region)", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(disc_mask, cmap="gray")
    axes[1].set_title("Stage 1 Output\n(Optic Disc Mask)", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(cup_mask, cmap="gray")
    axes[2].set_title("Stage 2 Output\n(Optic Cup Mask)", fontweight="bold")
    axes[2].axis("off")

    overlay = np.zeros((*roi_image.shape, 3), dtype=np.float32)
    overlay[:, :, 0] = roi_image
    overlay[:, :, 1] = roi_image
    overlay[:, :, 2] = roi_image

    disc_colored = disc_mask.astype(bool)
    overlay[disc_colored, 0] = 0.0
    overlay[disc_colored, 1] = 0.8
    overlay[disc_colored, 2] = 0.0

    cup_colored = cup_mask.astype(bool)
    overlay[cup_colored, 0] = 1.0
    overlay[cup_colored, 1] = 1.0
    overlay[cup_colored, 2] = 0.0

    overlay = np.clip(overlay, 0, 1)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay\n(Green=Disc, Yellow=Cup)", fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Enhanced K-Strange Segmentation Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved segmentation figure to {save_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def detect_optic_disc_roi(
    image: np.ndarray,
    min_area_ratio: float = 0.003,
    threshold_percentile: float = 90.0,
    padding_ratio: float = 0.45,
    crop_size: int = 200,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Detect a bright optic-disc ROI using a connected-component search.

    The optic disc is typically one of the brightest compact structures in a
    fundus image. This detector smooths the image, thresholds the bright tail,
    and picks the strongest connected component by a combination of area and
    mean intensity.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image for ROI extraction.")

    h, w = image.shape
    smoothed = cv2.GaussianBlur(image.astype(np.float32), (9, 9), 0)

    threshold_value = float(np.percentile(smoothed, threshold_percentile))
    bright_mask = (smoothed >= threshold_value).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask, connectivity=8
    )

    min_area = max(25, int(h * w * min_area_ratio))
    best_label = -1
    best_score = -1.0
    best_bbox = (0, 0, w, h)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        touches_border = x == 0 or y == 0 or (x + width) >= w or (y + height) >= h
        if touches_border:
            continue

        component_mask = labels == label
        mean_intensity = float(smoothed[component_mask].mean())
        cx, cy = centroids[label]

        center_dist = np.hypot(cx - (w / 2.0), cy - (h / 2.0))
        center_bonus = 1.0 / (1.0 + center_dist / max(h, w))
        bbox_area = max(1, width * height)
        compactness = area / bbox_area
        score = area * mean_intensity * center_bonus * compactness

        if compactness < 0.2:
            continue

        if score > best_score:
            best_score = score
            best_label = label
            best_bbox = (x, y, x + width, y + height)

    if best_label == -1:
        roi_h = max(1, int(h * 0.5))
        roi_w = max(1, int(w * 0.5))
        y1 = max(0, (h - roi_h) // 2)
        x1 = max(0, (w - roi_w) // 2)
        y2 = min(h, y1 + roi_h)
        x2 = min(w, x1 + roi_w)
        center_x = x1 + roi_w // 2
        center_y = y1 + roi_h // 2
    else:
        x1, y1, x2, y2 = best_bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h

        # Find the brightest point within the detected bbox to center the crop
        local_region = smoothed[y1:y2, x1:x2]
        if local_region.size == 0:
            center_x, center_y = w // 2, h // 2
        else:
            ly, lx = np.unravel_index(np.argmax(local_region), local_region.shape)
            center_x = x1 + int(lx)
            center_y = y1 + int(ly)

    # If crop_size is specified, extract a centered square patch of that size.
    if crop_size is not None and crop_size > 0:
        half = crop_size // 2
        cx = int(center_x)
        cy = int(center_y)

        x1c = cx - half
        y1c = cy - half
        x2c = cx + half
        y2c = cy + half

        # Clip to image bounds
        x1_clip = max(0, x1c)
        y1_clip = max(0, y1c)
        x2_clip = min(w, x2c)
        y2_clip = min(h, y2c)

        roi = image[y1_clip:y2_clip, x1_clip:x2_clip]

        # If crop is smaller than requested (near edges), pad to exact crop_size
        top = y1_clip - y1c if y1c < 0 else 0
        left = x1_clip - x1c if x1c < 0 else 0
        bottom = y2c - y2_clip if y2c > w else 0
        right = x2c - x2_clip if x2c > h else 0

        if any(v > 0 for v in (top, bottom, left, right)):
            roi = cv2.copyMakeBorder(
                roi,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )

        bbox = (max(0, x1c), max(0, y1c), min(w, x2c), min(h, y2c))
        center = (cx, cy)
        return roi, center, bbox

    # Fallback: return the padded bbox region if no crop requested
    roi = image[y1:y2, x1:x2]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    bbox = (x1, y1, x2, y2)
    return roi, center, bbox

    x1, y1, x2, y2 = best_bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, w, h

    roi = image[y1:y2, x1:x2]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    bbox = (x1, y1, x2, y2)
    return roi, center, bbox


def find_demo_image(project_root: Path) -> Path:
    """Find a real dataset image for the segmentation demo."""
    from preprocessing import find_first_image

    dataset_root = project_root / "dataset"
    search_roots = [
        dataset_root / "train",
        dataset_root / "val",
        dataset_root / "test",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for class_folder in sorted(root.iterdir()):
            if not class_folder.is_dir():
                continue
            image_path = find_first_image(class_folder)
            if image_path is not None:
                return image_path

    raise FileNotFoundError(
        f"No demo image found under {dataset_root}. Add an image to dataset/train, dataset/val, or dataset/test."
    )


def main() -> None:
    """Run the preprocessing -> ROI extraction -> segmentation demo flow."""
    from preprocessing import preprocess_image

    project_root = Path(__file__).resolve().parents[1]
    image_path = find_demo_image(project_root)
    output_plot = project_root / "outputs" / "plots" / f"segmentation_{image_path.stem}.png"
    print(f"Using demo image: {image_path}")

    print("\nStage 1 - Preprocessing")
    results = preprocess_image(str(image_path))
    print(f"  Normalized image shape: {results['normalized'].shape}")

    print("\nStage 2 - ROI Extraction")
    roi, _, _ = detect_optic_disc_roi(results["normalized"])
    print(f"  ROI shape: {roi.shape}")

    print("\nStage 3 - K-Strange Segmentation")
    disc_mask, cup_mask, _, _ = segment_disc_and_cup(roi)

    print("\nStage 4 - Segmentation Outputs")
    print(f"  Disc mask shape: {disc_mask.shape}")
    print(f"  Cup mask shape: {cup_mask.shape}")

    visualize_segmentation(roi, disc_mask, cup_mask, save_path=output_plot)


if __name__ == "__main__":
    main()
