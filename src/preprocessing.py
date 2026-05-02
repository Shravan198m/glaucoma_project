"""Image preprocessing pipeline for glaucoma fundus images."""

from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load a retinal fundus image from disk in BGR format (OpenCV default)."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def extract_green_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Extract the green channel from a BGR image."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image with shape (H, W, 3).")
    return image_bgr[:, :, 1]


def apply_gaussian_filter(
    image_gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian blur to reduce noise while preserving important edges."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number (3, 5, 7, ...).")
    return cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), sigma)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range for stable model training."""
    return image.astype(np.float32) / 255.0


def clahe_enhancement(image_gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE to improve local contrast under uneven fundus illumination."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def preprocess_image(
    image_path: str, kernel_size: int = 5, sigma: float = 1.0, use_clahe: bool = True
) -> Dict[str, np.ndarray]:
    """Run the full preprocessing pipeline for one fundus image."""
    image_bgr = load_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    green = extract_green_channel(image_bgr)
    enhanced = clahe_enhancement(green) if use_clahe else green.copy()
    filtered = apply_gaussian_filter(enhanced, kernel_size, sigma)
    normalized = normalize_image(filtered)

    return {
        "original_rgb": image_rgb,
        "green_channel": green,
        "clahe_enhanced": enhanced,
        "gaussian_filtered": filtered,
        "normalized": normalized,
        "pipeline_input": normalized,
    }


def visualize_preprocessing(
    results: Dict[str, np.ndarray], save_path: Optional[str] = None
) -> None:
    """Display preprocessing stages side by side and optionally save a figure."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # ── Build a proper GREEN image for display ──────────────────
    green_channel = results['green_channel']  # 2D array (H, W)

    # Create an RGB image where only the Green channel has data
    # Red=0, Green=actual data, Blue=0  → appears GREEN on screen
    h, w = green_channel.shape
    green_display = np.zeros((h, w, 3), dtype=np.uint8)
    green_display[:, :, 1] = green_channel   # Only fill G channel

    # ── Build GREEN versions of other stages too ─────────────────
    def to_green_rgb(gray_image):
        """Convert any grayscale image to green-tinted RGB display"""
        if gray_image.dtype != np.uint8:
            # Normalize to 0-255 first
            img = (gray_image * 255).astype(np.uint8)
        else:
            img = gray_image
        h, w = img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 1] = img  # Only green channel
        return rgb

    titles = [
        "Original RGB",
        "Green Channel",
        "CLAHE Enhanced",
        "Gaussian Filtered",
        "Normalized",
    ]
    images = [
        results["original_rgb"],                    # Normal RGB
        green_display,                              # Pure green
        to_green_rgb(results["clahe_enhanced"]),    # Green tint
        to_green_rgb(results["gaussian_filtered"]), # Green tint
        to_green_rgb(results["normalized"]),        # Green tint
    ]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)  # No cmap='gray' here!
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        ax.set_xlabel(f"Range: [{img.min():.3f}, {img.max():.3f}]", fontsize=8)

    plt.suptitle("Preprocessing Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path_obj}")

    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def _find_first_image(folder: Path) -> Optional[Path]:
    """Return first image path from a folder, or None if no images exist."""
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for file_path in sorted(folder.glob("*")):
        if file_path.is_file() and file_path.suffix.lower() in valid_ext:
            return file_path
    return None


def find_first_image(folder: Path) -> Optional[Path]:
    """Public helper that returns the first image path from a folder."""
    return _find_first_image(folder)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    default_folder = project_root / "dataset" / "train" / "normal"
    sample_image = _find_first_image(default_folder)

    if sample_image is None:
        print("No sample image found in dataset/train/normal.")
        print("Place images in the dataset folders and run again.")
    else:
        print("Running preprocessing pipeline...")
        print(f"Sample image: {sample_image}")
        output_plot = project_root / "outputs" / "plots" / "preprocessing.png"

        results = preprocess_image(str(sample_image), use_clahe=True)
        print(f"Original shape: {results['original_rgb'].shape}")
        print(f"Green channel shape: {results['green_channel'].shape}")
        print(
            "Normalized range: "
            f"[{results['normalized'].min():.3f}, {results['normalized'].max():.3f}]"
        )

        visualize_preprocessing(results, save_path=str(output_plot))
        print("Preprocessing complete.")
