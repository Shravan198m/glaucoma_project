"""
Canonical prediction service for the Glaucoma Detection API.

Orchestrates preprocessing, Enhanced K-Strange segmentation, CDR, and ResNet-50
using existing glaucoma_project modules. Single source of truth for inference.
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from cdr import compute_cdr, interpret_cdr
from config import MODELS_DIR, get_default_device
from model import create_resnet50_model
from predict import combine_decisions
from preprocessing import preprocess_image
from report_visualizer import generate_all_panels
from segmentation import detect_optic_disc_roi, segment_disc_and_cup

PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pth"


def _array_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _array_to_png_bytes(arr: np.ndarray, *, from_bgr: bool = False) -> bytes | None:
    """Encode an image array to PNG bytes (always correct RGB colour)."""
    if arr is None or arr.size == 0:
        return None
    if from_bgr and arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype != np.uint8:
        arr = _array_to_uint8(arr)
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _img_to_b64(
    path_or_array: Path | np.ndarray | None, *, from_bgr: bool = False
) -> str | None:
    if path_or_array is None:
        return None
    if isinstance(path_or_array, np.ndarray):
        raw = _array_to_png_bytes(path_or_array, from_bgr=from_bgr)
        return base64.b64encode(raw).decode() if raw else None
    p = Path(path_or_array)
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def _save_png(path: Path, arr: np.ndarray, *, from_bgr: bool = False) -> None:
    raw = _array_to_png_bytes(arr, from_bgr=from_bgr)
    if raw:
        path.write_bytes(raw)


def _build_segmentation_overlay(
    roi: np.ndarray, disc_mask: np.ndarray, cup_mask: np.ndarray
) -> np.ndarray:
    base = _array_to_uint8(roi)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    cup_fixed = (cup_mask.astype(np.uint8) * disc_mask.astype(np.uint8)).astype(np.uint8)
    overlay[disc_mask > 0] = [0, 220, 0]
    overlay[cup_fixed > 0] = [0, 220, 255]
    return overlay


def _map_final_prediction(final_diagnosis: dict[str, str]) -> str:
    label = final_diagnosis.get("label", "").upper()
    if "GLAUCOMA" in label and "BORDERLINE" not in label:
        return "Glaucoma"
    if "BORDERLINE" in label:
        return "Borderline"
    return "Normal"


def _pipeline_note(cdr: float, cnn_prediction: str, final_diagnosis: dict[str, str]) -> str:
    cdr_positive = cdr >= 0.6
    cnn_upper = cnn_prediction.upper()
    if cdr_positive and cnn_upper == "GLAUCOMA":
        return "CDR and CNN agree — glaucoma indicators detected."
    if (not cdr_positive) and cnn_upper == "NORMAL":
        return "CDR and CNN agree — within normal limits."
    return final_diagnosis.get("recommendation", "CDR and CNN disagree. Manual review needed.")


def compute_risk_level(cdr: float, prediction: str, confidence: float) -> str:
    cdr_interp = interpret_cdr(cdr)
    cdr_risk = cdr_interp["risk_level"]
    if prediction == "Glaucoma" and confidence >= 75 and cdr >= 0.6:
        return "High"
    if prediction == "Glaucoma" or cdr >= 0.6 or confidence < 60:
        return "Medium" if cdr_risk in ("Moderate Risk", "High Risk") else "Medium"
    if prediction == "Borderline":
        return "Medium"
    if cdr_risk == "Low Risk" and prediction == "Normal":
        return "Low"
    return "Medium"


def build_recommendations(prediction: str, cdr: float, risk_level: str) -> list[str]:
    recs: list[str] = [interpret_cdr(cdr)["recommendation"]]
    if risk_level == "Low":
        recs.extend([
            "Continue routine annual eye examinations.",
            "Maintain healthy lifestyle and monitor intraocular pressure if at risk.",
        ])
    elif risk_level == "Medium":
        recs.extend([
            "Schedule follow-up with an ophthalmologist within 3–6 months.",
            "Consider visual field testing and OCT imaging for confirmation.",
        ])
    else:
        recs.extend([
            "Urgent referral to a glaucoma specialist recommended.",
            "Intraocular pressure measurement and comprehensive eye exam required.",
        ])
    if prediction == "Glaucoma":
        recs.append("AI model indicates glaucomatous features — clinical correlation essential.")
    elif prediction == "Borderline":
        recs.append("CDR and CNN findings are inconclusive — specialist review required.")
    return recs


class PredictionService:
    """Loads the ResNet-50 model once and runs the full analysis pipeline."""

    def __init__(self, model_path: Path | str | None = None, device: str | None = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.device = torch.device(device or get_default_device())
        self.model = create_resnet50_model()
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        patient: dict[str, Any] | None = None,
        generate_panels: bool = True,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        report_id = f"GLC-{uuid.uuid4().hex[:8].upper()}"

        prep = preprocess_image(str(image_path), use_clahe=True)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        original_rgb = prep["original_rgb"]
        roi, _center, _bbox = detect_optic_disc_roi(prep["normalized"])
        disc_mask, cup_mask, _, _ = segment_disc_and_cup(roi)
        cup_fixed = (cup_mask.astype(np.uint8) * disc_mask.astype(np.uint8)).astype(np.uint8)

        cdr_val, cdr_details = compute_cdr(disc_mask, cup_fixed)
        cdr_interp = interpret_cdr(cdr_val)

        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = PREDICT_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probability = float(self.model(input_tensor).item())

        cnn_prediction = "GLAUCOMA" if probability >= 0.5 else "NORMAL"
        confidence = probability if cnn_prediction == "GLAUCOMA" else (1.0 - probability)
        confidence_pct = round(confidence * 100.0, 2)

        final_diagnosis = combine_decisions(cdr_val, cnn_prediction, probability)
        prediction = _map_final_prediction(final_diagnosis)
        final_note = _pipeline_note(cdr_val, cnn_prediction, final_diagnosis)
        risk_level = compute_risk_level(cdr_val, prediction, confidence_pct)
        recommendations = build_recommendations(prediction, cdr_val, risk_level)

        disc_area = int(cdr_details.get("disc_area_pixels", 0))
        cup_area = int(cdr_details.get("cup_area_pixels", 0))
        rim_area = max(0, disc_area - cup_area)

        seg_overlay = _build_segmentation_overlay(roi, disc_mask, cup_fixed)
        vessel_rgb = cv2.cvtColor(_array_to_uint8(prep["green_channel"]), cv2.COLOR_GRAY2BGR)

        if generate_panels:
            panel_images = generate_all_panels(
                prep=prep,
                roi=roi,
                disc_mask=disc_mask,
                cup_mask=cup_fixed,
                cdr=cdr_val,
                cdr_details=cdr_details,
                cdr_status=cdr_interp["status"],
                cnn_label=cnn_prediction.title(),
                cnn_confidence=confidence_pct,
                cnn_probability=probability,
                final_label=final_diagnosis["label"],
                final_confidence=confidence_pct,
                final_note=final_note,
                final_diagnosis=final_diagnosis,
            )
        else:
            panel_images = {
                "preprocessing_image": None,
                "segmentation_panel_image": None,
                "cdr_report_image": None,
                "pipeline_summary_image": None,
                "final_composite_image": None,
                "resnet_result_image": None,
            }

        images = {
            "original": _img_to_b64(original_rgb),
            "optic_disc": _img_to_b64(seg_overlay, from_bgr=True),
            "optic_cup": _img_to_b64(seg_overlay, from_bgr=True),
            "segmentation": _img_to_b64(seg_overlay, from_bgr=True),
            "vessels": _img_to_b64(vessel_rgb, from_bgr=True),
            "green": _img_to_b64(_array_to_uint8(prep["green_channel"])),
            "gaussian": _img_to_b64(_array_to_uint8(prep["gaussian_filtered"])),
            "normalized": _img_to_b64(_array_to_uint8(prep["normalized"])),
            **panel_images,
        }

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _save_png(out / "original.png", original_rgb)
            _save_png(out / "segmentation.png", seg_overlay, from_bgr=True)
            _save_png(out / "vessels.png", vessel_rgb, from_bgr=True)
            if generate_panels:
                for key in panel_images:
                    if panel_images[key]:
                        raw = base64.b64decode(panel_images[key])
                        (out / f"{key}.png").write_bytes(raw)

        return {
            "report_id": report_id,
            "prediction": prediction,
            "confidence_score": confidence_pct,
            "cup_disc_ratio": round(cdr_val, 4),
            "risk_level": risk_level,
            "disc_area": disc_area,
            "cup_area": cup_area,
            "rim_area": rim_area,
            "cdr_status": cdr_interp["status"],
            "final_diagnosis": final_diagnosis,
            "cnn_prediction": cnn_prediction,
            "recommendations": recommendations,
            "patient": patient or {},
            "images": images,
            "segmentation_images": {
                "original": images["original"],
                "optic_disc": images["optic_disc"],
                "optic_cup": images["optic_cup"],
                "segmentation": images["segmentation"],
                "vessels": images["vessels"],
            },
            "preprocessing_image": images["preprocessing_image"],
            "segmentation_panel_image": images["segmentation_panel_image"],
            "cdr_report_image": images["cdr_report_image"],
            "pipeline_summary_image": images["pipeline_summary_image"],
            "final_composite_image": images["final_composite_image"],
            "resnet_result_image": images["resnet_result_image"],
            "resnet_probability": round(probability, 4),
            "resnet_model": "ResNet-50",
            "original_image": images["original"],
            "segmentation_image": images["segmentation"],
            "stages": {
                "original": images["original"],
                "green": images["green"],
                "gaussian": images["gaussian"],
                "normalized": images["normalized"],
                "segmented": images["segmentation"],
            },
            "diagnosis": prediction,
            "confidence": confidence_pct,
            "cdr": round(cdr_val, 4),
            "_pipeline_data": {
                "prep": prep,
                "roi": roi,
                "disc_mask": disc_mask,
                "cup_fixed": cup_fixed,
                "cdr_val": cdr_val,
                "cdr_details": cdr_details,
                "cdr_status": cdr_interp["status"],
                "cnn_prediction": cnn_prediction,
                "confidence_pct": confidence_pct,
                "probability": probability,
                "final_diagnosis": final_diagnosis,
                "final_note": final_note,
            }
        }
