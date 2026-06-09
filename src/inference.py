"""Run CNN inference for a single image and save results.

Saves JSON with label/probability/confidence and an annotated PNG.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont

from segmentation import find_demo_image
from model import create_resnet50_model
import torch


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


def _predict_single_image(image_path: Path, model_path: Path, threshold: float = 0.5) -> dict:
    """Load model and run inference on a single image."""
    from PIL import Image
    from torchvision import transforms
    
    device = torch.device('cpu')
    model = create_resnet50_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(image_path).convert('RGB')
    image_tensor = cast(torch.Tensor, transform(pil_image))
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        probability = float(model(input_tensor).item())
    
    label = "Glaucoma" if probability >= threshold else "Normal"
    confidence = probability if label == "Glaucoma" else 1.0 - probability
    
    return {
        "label": label,
        "probability": probability,
        "confidence": confidence,
        "confidence_percent": round(confidence * 100.0, 2),
        "threshold": threshold,
    }


def main(threshold: float = 0.5) -> None:
    project_root = PROJECT_ROOT
    image_path = find_demo_image(project_root)
    print(f"Demo image: {image_path}")

    model_path = project_root / "outputs" / "models" / "best_model.pth"
    print("Running ResNet inference...")
    result = _predict_single_image(Path(image_path), model_path, threshold=threshold)

    print(f"Label: {result['label']}")
    print(f"Probability: {result['probability']:.6f}")
    print(f"Confidence: {result['confidence_percent']}%")
    print(f"Decision threshold: {result['threshold']:.4f}")

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
