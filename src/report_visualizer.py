"""Generate clinical pipeline panel images for API, PDF, and frontend."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cdr import display_cdr_report
from preprocessing import visualize_preprocessing
from segmentation import visualize_segmentation


def _fig_to_b64(fig: plt.Figure, facecolor: str | None = None) -> str:
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {"format": "png", "dpi": 96, "bbox_inches": "tight"}
    if facecolor:
        kwargs["facecolor"] = facecolor
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _temp_panel(render_fn, *args, **kwargs) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        render_fn(*args, save_path=path, **kwargs)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        Path(path).unlink(missing_ok=True)


def render_preprocessing_panel(prep: dict[str, np.ndarray]) -> str:
    return _temp_panel(visualize_preprocessing, prep)


def render_segmentation_panel(
    roi: np.ndarray, disc_mask: np.ndarray, cup_mask: np.ndarray
) -> str:
    return _temp_panel(visualize_segmentation, roi, disc_mask, cup_mask)


def render_cdr_panel(
    roi: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    cdr_details: dict[str, Any],
) -> str:
    return _temp_panel(display_cdr_report, roi, disc_mask, cup_mask, cdr, cdr_details)


def render_pipeline_summary(
    original_rgb: np.ndarray,
    roi: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    cdr_status: str,
    cnn_label: str,
    cnn_confidence: float,
    final_label: str,
    final_confidence: float,
    note: str,
) -> str:
    """Three-panel summary: input, ROI+segmentation, clinical text."""
    overlay = np.stack([roi] * 3, axis=-1).astype(np.float32)
    if roi.dtype != np.float32:
        overlay = overlay / 255.0

    disc_coords = np.where(disc_mask > 0)
    cup_coords = np.where(cup_mask > 0)
    overlay[disc_coords[0], disc_coords[1], 0] = 0.0
    overlay[disc_coords[0], disc_coords[1], 1] = 0.8
    overlay[disc_coords[0], disc_coords[1], 2] = 0.0
    overlay[cup_coords[0], cup_coords[1], 0] = 1.0
    overlay[cup_coords[0], cup_coords[1], 1] = 1.0
    overlay[cup_coords[0], cup_coords[1], 2] = 0.0

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), facecolor="#0A2540")
    for ax in axes:
        ax.set_facecolor("#0A2540")
        
    axes[0].imshow(original_rgb)
    axes[0].set_title("Input", fontweight="bold", fontsize=9, color="white")
    axes[0].axis("off")

    seg_overlay = np.clip(overlay, 0, 1)
    axes[1].imshow(seg_overlay)
    axes[1].set_title("Enhanced K-Strange", fontweight="bold", fontsize=9, color="white")
    axes[1].axis("off")

    axes[2].imshow(original_rgb)
    axes[2].set_title(f"ResNet-50\n{cnn_label} ({cnn_confidence:.1f}%)", fontweight="bold", fontsize=9, color="white")
    axes[2].axis("off")

    axes[3].bar(["CDR"], [cdr], color="#EF4444" if cdr >= 0.6 else "#10B981", width=0.4)
    axes[3].axhline(y=0.6, color="#F59E0B", linestyle="--", linewidth=1.5)
    axes[3].set_ylim(0, 1)
    axes[3].set_title(f"CDR: {cdr:.3f}", fontweight="bold", fontsize=9, color="white")
    axes[3].tick_params(colors="white")
    for spine in axes[3].spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.08))


    axes[4].axis("off")
    text = (
        f"FINAL\n{final_label}\n\n"
        f"CDR: {cdr_status}\n"
        f"CNN: {cnn_label}\n"
        f"{note}"
    )
    axes[4].text(
        0.05,
        0.95,
        text,
        transform=axes[4].transAxes,
        fontsize=9,
        va="top",
        fontfamily="monospace",
        color="white",
        bbox=dict(boxstyle="round", facecolor="#0F2D4D", edgecolor="#00C2FF", linewidth=1.5),
    )

    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.85, wspace=0.45)
    return _fig_to_b64(fig)


def render_final_composite(
    prep: dict[str, np.ndarray],
    roi: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    cnn_pred: str,
    confidence: float,
    final_diagnosis: dict[str, str],
) -> str:
    """Full 12-panel diagnostic composite matching offline CLI output."""

    def to_green(gray: np.ndarray) -> np.ndarray:
        if gray.dtype != np.uint8:
            img = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        else:
            img = gray.copy()
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        rgb[:, :, 1] = img
        return rgb

    import matplotlib.patches as mpatches

    original = prep["original_rgb"]
    green_channel = prep["green_channel"]
    clahe_image = prep["clahe_enhanced"]
    normalized_image = prep["normalized"]
    cup_mask_fixed = (cup_mask.astype(np.uint8) * disc_mask.astype(np.uint8)).astype(np.uint8)

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor="#0A2540")
    for r_axes in axes:
        for ax in r_axes:
            ax.set_facecolor("#0A2540")

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
        facecolor="#0F2D4D",
        labelcolor="white",
    )
    axes[1, 3].set_title("Segmentation overlay", color="white", fontweight="bold")
    axes[1, 3].axis("off")

    ax5 = axes[2, 0]
    cdr_color = "#EF4444" if cdr >= 0.6 else "#10B981"
    ax5.bar(["CDR"], [cdr], color=cdr_color, width=0.4, edgecolor="white")
    ax5.axhline(y=0.6, color="#F59E0B", linestyle="--", linewidth=2, label="Threshold (0.6)")
    ax5.set_ylim(0, 1.0)
    ax5.set_title("Cup-to-Disc Ratio", color="white", fontweight="bold")
    ax5.set_facecolor("#0F2D4D")
    ax5.tick_params(colors="white")
    ax5.spines["bottom"].set_color("white")
    ax5.spines["left"].set_color("white")
    ax5.text(0, cdr + 0.02, f"{cdr:.3f}", ha="center", color="white", fontweight="bold", fontsize=14)
    ax5.legend(facecolor="#0F2D4D", labelcolor="white")

    ax6 = axes[2, 1]
    normal_prob = 100.0 - confidence if cnn_pred == "GLAUCOMA" else confidence
    glaucoma_prob = confidence if cnn_pred == "GLAUCOMA" else 100.0 - confidence
    ax6.bar(["Normal", "Glaucoma"], [normal_prob, glaucoma_prob], color=["#10B981", "#EF4444"], edgecolor="white")
    ax6.set_ylim(0, 100)
    ax6.set_title("CNN Prediction Confidence", color="white", fontweight="bold")
    ax6.set_facecolor("#0F2D4D")
    ax6.tick_params(colors="white")
    ax6.spines["bottom"].set_color("white")
    ax6.spines["left"].set_color("white")
    ax6.set_ylabel("Confidence (%)", color="white")

    axes[2, 2].axis("off")
    axes[2, 3].axis("off")

    result_color = final_diagnosis.get("color", "white")
    if result_color == "red":
        result_color = "#EF4444"
    elif result_color == "green":
        result_color = "#10B981"
    elif result_color == "orange":
        result_color = "#F59E0B"

    result_text = (
        "DIAGNOSIS REPORT\n"
        + ("-" * 30)
        + "\n\n"
        + f"CDR Value:  {cdr:.3f}\n"
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
        bbox=dict(boxstyle="round", facecolor="#0F2D4D", edgecolor=result_color, linewidth=3),
    )

    plt.tight_layout()
    return _fig_to_b64(fig, facecolor="#0A2540")


def render_resnet_panel(
    original_rgb: np.ndarray,
    cnn_label: str,
    confidence_pct: float,
    probability: float,
    final_label: str,
) -> str:
    """ResNet-50 classification result annotated on the fundus image."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(_to_uint8(original_rgb))
    draw = ImageDraw.Draw(img)
    w, h = img.size

    is_glaucoma = "glaucoma" in cnn_label.lower()
    box_color = (239, 68, 68) if is_glaucoma else (16, 185, 129)
    text = (
        f"ResNet-50 Classification\n"
        f"CNN: {cnn_label.upper()}\n"
        f"Confidence: {confidence_pct:.1f}%\n"
        f"Probability: {probability:.4f}\n"
        f"Fused: {final_label}"
    )

    pad = 12
    box_h = 120
    draw.rectangle([10, 10, min(w - 10, 360), 10 + box_h], fill=(10, 37, 64), outline=(0, 194, 255), width=3)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text((10 + pad, 10 + pad), text, fill=(255, 255, 255), font=font, spacing=4)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()



def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def generate_all_panels(
    prep: dict[str, np.ndarray],
    roi: np.ndarray,
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    cdr: float,
    cdr_details: dict[str, Any],
    cdr_status: str,
    cnn_label: str,
    cnn_confidence: float,
    cnn_probability: float,
    final_label: str,
    final_confidence: float,
    final_note: str,
    final_diagnosis: dict[str, str],
) -> dict[str, str]:
    """Return base64 PNG strings for every clinical pipeline panel."""
    cup_fixed = (cup_mask.astype(np.uint8) * disc_mask.astype(np.uint8)).astype(np.uint8)
    cnn_pred_upper = "GLAUCOMA" if "glaucoma" in cnn_label.lower() else "NORMAL"

    return {
        "preprocessing_image": render_preprocessing_panel(prep),
        "segmentation_panel_image": render_segmentation_panel(roi, disc_mask, cup_fixed),
        "cdr_report_image": render_cdr_panel(roi, disc_mask, cup_fixed, cdr, cdr_details),
        "pipeline_summary_image": render_pipeline_summary(
            prep["original_rgb"],
            roi,
            disc_mask,
            cup_fixed,
            cdr,
            cdr_status,
            cnn_label,
            cnn_confidence,
            final_label,
            final_confidence,
            final_note,
        ),
        "final_composite_image": render_final_composite(
            prep,
            roi,
            disc_mask,
            cup_fixed,
            cdr,
            cnn_pred_upper,
            cnn_confidence,
            final_diagnosis,
        ),
        "resnet_result_image": render_resnet_panel(
            prep["original_rgb"],
            cnn_label,
            cnn_confidence,
            cnn_probability,
            final_label,
        ),
    }
