"""Run CNN inference for a single image and save results.

Saves JSON with label/probability/confidence and an annotated PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate import predict_single_image
from segmentation import find_demo_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _annotate_and_save(image_path: Path, result: dict, save_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()

    text_lines = [f"Label: {result['label']}", f"Prob: {result['probability']:.4f}", f"Confidence: {result['confidence_percent']}%"]
    x, y = 10, 10
    for line in text_lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += 20

    save_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def main() -> None:
    project_root = PROJECT_ROOT
    image_path = find_demo_image(project_root)
    print(f"Demo image: {image_path}")

    model_path = project_root / "outputs" / "models" / "best_model.pth"
    print("Running ResNet inference...")
    result = predict_single_image(str(image_path), str(model_path))

    print(f"Label: {result['label']}")
    print(f"Probability: {result['probability']:.6f}")
    print(f"Confidence: {result['confidence_percent']}%")

    results_root = project_root / "outputs" / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    json_path = results_root / f"resnet_{image_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved JSON result to: {json_path}")

    annotated_path = results_root / f"resnet_{image_path.stem}.png"
    _annotate_and_save(image_path, result, annotated_path)
    print(f"Saved annotated image to: {annotated_path}")


if __name__ == "__main__":
    main()
