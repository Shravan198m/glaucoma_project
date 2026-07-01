from pathlib import Path
import csv
import sys
import cv2
import numpy as np

project = Path(r"c:/Users/svmoo/OneDrive/Documents/GLAUCOMA/glaucoma_project")
sys.path.insert(0, str(project / "src"))
from preprocessing import preprocess_image


def to_green_bgr(image_u8: np.ndarray) -> np.ndarray:
    """Convert single-channel image to a 3-channel image with only green populated."""
    if image_u8.ndim != 2:
        raise ValueError("Expected a 2D single-channel image.")
    h, w = image_u8.shape
    green_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    green_bgr[:, :, 1] = image_u8
    return green_bgr

in_root = project / "dataset"
out_root = project / "outputs" / "preprocessed"
out_root.mkdir(parents=True, exist_ok=True)
summary_path = out_root / "summary.csv"

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
processed = 0
failed = 0
rows = []

for split in ["train", "val", "test"]:
    split_dir = in_root / split
    if not split_dir.exists():
        continue

    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    for cls_dir in sorted(class_dirs):
        out_dir = out_root / split / cls_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(cls_dir.iterdir()):
            if (not img_path.is_file()) or (img_path.suffix.lower() not in valid_ext):
                continue
            try:
                res = preprocess_image(str(img_path), use_clahe=True)
                norm = res["normalized"]
                norm_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
                norm_green_bgr = to_green_bgr(norm_u8)
                out_file = out_dir / (img_path.stem + "_preprocessed.png")
                cv2.imwrite(str(out_file), norm_green_bgr)
                processed += 1
                norm_min = str(round(float(norm.min()), 6))
                norm_max = str(round(float(norm.max()), 6))
                rows.append([
                    split,
                    cls_dir.name,
                    str(img_path.relative_to(project)),
                    str(out_file.relative_to(project)),
                    norm_min,
                    norm_max,
                ])
            except Exception as e:
                failed += 1
                rows.append([
                    split,
                    cls_dir.name,
                    str(img_path.relative_to(project)),
                    "ERROR",
                    "",
                    "",
                ])
                print("Failed: " + str(img_path) + " -> " + str(e))

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "class", "input", "output", "norm_min", "norm_max"])
    writer.writerows(rows)

print("Batch preprocessing finished. processed=" + str(processed) + ", failed=" + str(failed))
print("Summary CSV: " + str(summary_path))
