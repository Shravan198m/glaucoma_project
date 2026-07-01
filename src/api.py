"""
Glaucoma Detection REST API — canonical backend for frontend and CLI clients.
"""

from __future__ import annotations

import base64
import json
import numpy as np
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# Ensure sibling modules (config, predict, etc.) resolve when loaded as src.api
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import OUTPUTS_DIR, get_default_device
from pdf_service import generate_medical_pdf
from prediction_service import DEFAULT_MODEL_PATH, PredictionService

REPORTS_DIR = OUTPUTS_DIR / "reports"
RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# On-disk image files for GET /results/{job_id} (avoids huge sessionStorage payloads)
RESULT_IMAGE_FILES: dict[str, str] = {
    "preprocessing_image": "preprocessing_image.png",
    "segmentation_panel_image": "segmentation_panel_image.png",
    "cdr_report_image": "cdr_report_image.png",
    "pipeline_summary_image": "pipeline_summary_image.png",
    "final_composite_image": "final_composite_image.png",
    "resnet_result_image": "resnet_result_image.png",
    "original_image": "original.png",
    "segmentation_image": "segmentation.png",
    "vessel_image": "vessels.png",
}

EVAL_PLOTS = frozenset({
    "roc_curve.png",
    "confusion_matrix.png",
    "preprocessing_normal_000001.png",
    "segmentation_normal_000001.png",
    "cdr_normal_000001.png",
})

TRAINING_PLOTS = frozenset({
    "training_history_optimized.png",
})

TRAINING_DIR = OUTPUTS_DIR / "training_results"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

app = FastAPI(
    title="Glaucoma Detection API",
    description="REST API for glaucoma detection from fundus images",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: PredictionService | None = None


def get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService(
            model_path=DEFAULT_MODEL_PATH,
            device=get_default_device(),
        )
    return _service


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_exists: bool
    device: str
    version: str = "2.1.0"


class PredictionResult(BaseModel):
    job_id: str
    report_id: str
    prediction: str
    confidence_score: float
    cup_disc_ratio: float
    risk_level: str
    disc_area: int
    cup_area: int
    rim_area: int
    cdr_status: str = ""
    recommendations: list[str]
    segmentation_images: dict[str, str | None]
    optic_disc_image: str | None = None
    optic_cup_image: str | None = None
    segmentation_image: str | None = None
    original_image: str | None = None
    vessel_image: str | None = None
    preprocessing_image: str | None = None
    segmentation_panel_image: str | None = None
    cdr_report_image: str | None = None
    pipeline_summary_image: str | None = None
    final_composite_image: str | None = None
    resnet_result_image: str | None = None
    resnet_probability: float = 0.0
    resnet_model: str = "ResNet-50"
    pdf_url: str
    pdf_status: str = "ready"
    stages: dict[str, str | None] = Field(default_factory=dict)
    diagnosis: str
    confidence: float
    cdr: float
    final_diagnosis_label: str = ""
    cnn_prediction: str = ""
    patient_name: str = ""
    patient_age: str = ""
    patient_id: str = ""



@app.on_event("startup")
async def startup_event() -> None:
    try:
        get_service()
        print(f"Model loaded from {DEFAULT_MODEL_PATH}")
    except Exception as exc:
        print(f"Warning: model not loaded at startup: {exc}")


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    loaded = _service is not None
    return HealthResponse(
        status="ok" if loaded and DEFAULT_MODEL_PATH.exists() else "model_not_loaded",
        model_loaded=loaded,
        model_path=str(DEFAULT_MODEL_PATH),
        model_exists=DEFAULT_MODEL_PATH.exists(),
        device=get_default_device(),
    )


def _file_to_b64(path: Path) -> str | None:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def _save_result_metadata(result_dir: Path, payload: dict[str, Any]) -> None:
    """Persist scalar result fields; images are loaded from disk on demand."""
    slim = {
        k: v
        for k, v in payload.items()
        if k
        not in (
            "segmentation_images",
            "stages",
            "preprocessing_image",
            "segmentation_panel_image",
            "cdr_report_image",
            "pipeline_summary_image",
            "final_composite_image",
            "resnet_result_image",
            "original_image",
            "segmentation_image",
            "vessel_image",
            "optic_disc_image",
            "optic_cup_image",
        )
    }
    (result_dir / "metadata.json").write_text(json.dumps(slim), encoding="utf-8")


def _result_file_url(base_url: str, job_id: str, filename: str) -> str:
    return f"{base_url}/results/{job_id}/files/{filename}"


def _attach_image_urls(payload: dict[str, Any], result_dir: Path, base_url: str, job_id: str) -> None:
    """Prefer image URLs (fast) over base64 when serving saved results from disk."""
    for api_key, filename in RESULT_IMAGE_FILES.items():
        file_path = result_dir / filename
        if file_path.exists():
            payload[api_key] = _result_file_url(base_url, job_id, filename)
        elif payload.get(api_key):
            continue  # keep existing base64 from POST response
        else:
            payload[api_key] = None

    seg = payload.get("segmentation_image")
    orig = payload.get("original_image")
    vessels = payload.get("vessel_image")
    payload["segmentation_images"] = {
        "original": orig,
        "optic_disc": seg,
        "optic_cup": seg,
        "segmentation": seg,
        "vessels": vessels,
    }
    payload["optic_disc_image"] = seg
    payload["optic_cup_image"] = seg
    payload["stages"] = {
        "original": orig,
        "segmented": seg,
    }


def _ensure_resnet_panel(result_dir: Path, payload: dict[str, Any]) -> None:
    """Regenerate ResNet panel PNG if missing (legacy jobs or interrupted saves)."""
    out_path = result_dir / "resnet_result_image.png"
    if out_path.exists():
        return
    orig_path = result_dir / "original.png"
    if not orig_path.exists():
        return
    try:
        import cv2
        from report_visualizer import render_resnet_panel

        bgr = cv2.imread(str(orig_path))
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cnn = payload.get("cnn_prediction", "NORMAL")
        conf = float(payload.get("confidence_score", 0))
        prob = float(payload.get("resnet_probability", 0.5))
        if prob == 0.5 and cnn.upper() == "GLAUCOMA":
            prob = conf / 100.0
        elif prob == 0.5 and cnn.upper() == "NORMAL":
            prob = 1.0 - conf / 100.0
        b64 = render_resnet_panel(
            rgb,
            cnn.title(),
            conf,
            prob,
            payload.get("final_diagnosis_label", payload.get("prediction", "")),
        )
        out_path.write_bytes(base64.b64decode(b64))
    except Exception as exc:
        print(f"ResNet panel regeneration failed: {exc}")


def _load_result_from_disk(job_id: str, base_url: str) -> dict[str, Any]:
    safe_id = job_id.replace("..", "").replace("/", "")
    result_dir = RESULTS_DIR / safe_id
    if not result_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail="No saved analysis found. Go to Analysis and upload a fundus image.",
        )

    meta_path = result_dir / "metadata.json"
    if meta_path.exists():
        payload: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        patient = payload.get("patient", {})
        if "patient_name" not in payload:
            payload["patient_name"] = payload.get("patient_name", patient.get("name", ""))
        if "patient_age" not in payload:
            payload["patient_age"] = payload.get("patient_age", patient.get("age", ""))
        if "patient_id" not in payload:
            payload["patient_id"] = payload.get("patient_id", patient.get("id", ""))
    elif (result_dir / "original.png").exists():
        # Legacy job folder (before metadata.json) — images only, re-analyze for full metrics
        payload = {
            "job_id": safe_id,
            "report_id": "",
            "prediction": "Unknown",
            "confidence_score": 0.0,
            "cup_disc_ratio": 0.0,
            "risk_level": "Medium",
            "disc_area": 0,
            "cup_area": 0,
            "rim_area": 0,
            "cdr_status": "",
            "recommendations": [
                "Partial results recovered from disk.",
                "Please run a new analysis for full clinical metrics and pipeline panels.",
            ],
            "diagnosis": "Unknown",
            "confidence": 0.0,
            "cdr": 0.0,
            "final_diagnosis_label": "",
            "cnn_prediction": "",
            "pdf_url": "",
            "patient_name": "",
            "patient_age": "",
            "patient_id": "",
        }
    else:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please upload a fundus image on the Analysis page.",
        )

    _ensure_resnet_panel(result_dir, payload)
    _attach_image_urls(payload, result_dir, base_url, safe_id)
    if payload.get("report_id"):
        payload["pdf_url"] = f"{base_url}/reports/{payload['report_id']}.pdf"
        pdf_file = REPORTS_DIR / f"{payload['report_id']}.pdf"
        if payload.get("pdf_status") not in ("ready", "failed"):
            payload["pdf_status"] = "ready" if pdf_file.exists() else "generating"
    
    # Ensure patient fields are always set in loaded payload
    payload["patient_name"] = payload.get("patient_name", "")
    payload["patient_age"] = payload.get("patient_age", "")
    payload["patient_id"] = payload.get("patient_id", "")
    return payload


def _build_api_response(result: dict[str, Any], base_url: str, job_id: str) -> dict[str, Any]:
    seg = result["segmentation_images"]
    final = result.get("final_diagnosis", {})
    patient = result.get("patient", {})
    return {
        "job_id": job_id,
        "report_id": result["report_id"],
        "prediction": result["prediction"],
        "confidence_score": result["confidence_score"],
        "cup_disc_ratio": result["cup_disc_ratio"],
        "risk_level": result["risk_level"],
        "disc_area": result["disc_area"],
        "cup_area": result["cup_area"],
        "rim_area": result["rim_area"],
        "cdr_status": result.get("cdr_status", ""),
        "recommendations": result["recommendations"],
        "segmentation_images": seg,
        "optic_disc_image": seg.get("optic_disc"),
        "optic_cup_image": seg.get("optic_cup"),
        "segmentation_image": seg.get("segmentation"),
        "original_image": seg.get("original"),
        "vessel_image": seg.get("vessels"),
        "preprocessing_image": result.get("preprocessing_image"),
        "segmentation_panel_image": result.get("segmentation_panel_image"),
        "cdr_report_image": result.get("cdr_report_image"),
        "pipeline_summary_image": result.get("pipeline_summary_image"),
        "final_composite_image": result.get("final_composite_image"),
        "resnet_result_image": result.get("resnet_result_image"),
        "resnet_probability": result.get("resnet_probability", 0.0),
        "resnet_model": result.get("resnet_model", "ResNet-50"),
        "pdf_url": f"{base_url}/reports/{result['report_id']}.pdf",
        "pdf_status": result.get("pdf_status", "ready"),
        "stages": result["stages"],
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "cdr": result["cdr"],
        "final_diagnosis_label": final.get("label", ""),
        "cnn_prediction": result.get("cnn_prediction", ""),
        "patient_name": result.get("patient_name", patient.get("name", "")),
        "patient_age": result.get("patient_age", patient.get("age", "")),
        "patient_id": result.get("patient_id", patient.get("id", "")),
    }



def _generate_panels_and_pdf_task(
    result: dict[str, Any],
    pipeline_data: dict[str, Any] | None,
    pdf_path: Path,
    result_dir: Path,
) -> None:
    try:
        if pipeline_data:
            from report_visualizer import generate_all_panels
            panel_images = generate_all_panels(
                prep=pipeline_data["prep"],
                roi=pipeline_data["roi"],
                disc_mask=pipeline_data["disc_mask"],
                cup_mask=pipeline_data["cup_fixed"],
                cdr=pipeline_data["cdr_val"],
                cdr_details=pipeline_data["cdr_details"],
                cdr_status=pipeline_data["cdr_status"],
                cnn_label=pipeline_data["cnn_prediction"].title(),
                cnn_confidence=pipeline_data["confidence_pct"],
                cnn_probability=pipeline_data["probability"],
                final_label=pipeline_data["final_diagnosis"]["label"],
                final_confidence=pipeline_data["confidence_pct"],
                final_note=pipeline_data["final_note"],
                final_diagnosis=pipeline_data["final_diagnosis"],
            )
            for key, val in panel_images.items():
                if val:
                    raw = base64.b64decode(val)
                    (result_dir / f"{key}.png").write_bytes(raw)

        generate_medical_pdf(result, pdf_path, result_dir=result_dir)
        meta_path = result_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["pdf_status"] = "ready"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except Exception as exc:
        print(f"Panel/PDF generation failed: {exc}")
        meta_path = result_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["pdf_status"] = "failed"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")


def validate_fundus_image(image_bgr: np.ndarray) -> tuple[bool, str]:
    """Validate if the BGR image is a retinal fundus photograph."""
    import cv2
    if image_bgr is None or image_bgr.size == 0:
        return False, "Empty or invalid image data."

    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        return False, "Invalid image: The uploaded file is a grayscale scan or lacks color channels. Please upload a clear color fundus photograph."

    h, w, c = image_bgr.shape

    # Downsample image for fast color analysis (100x100 is extremely fast and sufficient)
    small = cv2.resize(image_bgr, (100, 100))

    # Split into B, G, R channels
    b, g, r = small[:, :, 0].astype(np.float32), small[:, :, 1].astype(np.float32), small[:, :, 2].astype(np.float32)

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    # 1. Check for completely black or extremely dark images
    if r_mean < 5.0 and g_mean < 5.0 and b_mean < 5.0:
        return False, "Invalid image: The uploaded photograph is too dark or completely black."

    # 2. Check for completely white or overexposed images
    if r_mean > 245.0 and g_mean > 245.0 and b_mean > 245.0:
        return False, "Invalid image: The uploaded photograph is overexposed or completely white."

    # 3. Grayscale / low saturation check (e.g., document scans, receipts, text, grayscale scans)
    color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
    if color_diff < 8.0:
        return False, "Invalid image: The uploaded file lacks color saturation and appears to be a document, receipt, or grayscale scan. Please upload a clear color fundus photograph."

    # 4. Red-to-Blue ratio check (fundus images are highly dominated by red/orange hues)
    red_blue_ratio = r_mean / (b_mean + 1e-5)
    if red_blue_ratio < 1.2:
        return False, "Invalid image: The uploaded photograph color profile does not match a retinal fundus image (lacks red/orange dominance). Please upload a valid color fundus photograph."

    # 5. Red-to-Green ratio check (red must be brighter than green in fundus photography)
    if r_mean * 1.1 < g_mean:
        return False, "Invalid image: The uploaded photograph color profile does not match a retinal fundus image (green channel is brighter than red). Please upload a valid color fundus photograph."

    return True, "Valid fundus image."


@app.post("/predict", response_model=PredictionResult)
async def predict_glaucoma(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_name: str = Form(""),
    patient_age: str = Form(""),
    patient_id: str = Form(""),
) -> PredictionResult:
    if _service is None:
        try:
            get_service()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc

    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = str(uuid.uuid4())[:8]
    patient = {"name": patient_name, "age": patient_age, "id": patient_id}
    temp_path: Path | None = None

    try:
        image_data = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_data)
            temp_path = Path(tmp.name)

        # Validate that the image is a valid retinal fundus photograph
        import cv2
        img_bgr = cv2.imread(str(temp_path))
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image or is corrupted.")

        is_valid, msg = validate_fundus_image(img_bgr)
        if not is_valid:
            raise HTTPException(status_code=400, detail=msg)

        result_dir = RESULTS_DIR / job_id
        result = get_service().predict(temp_path, output_dir=result_dir, patient=patient, generate_panels=False)
        pipeline_data = result.pop("_pipeline_data", None)

        pdf_path = REPORTS_DIR / f"{result['report_id']}.pdf"
        result["pdf_status"] = "generating"
        background_tasks.add_task(_generate_panels_and_pdf_task, result, pipeline_data, pdf_path, result_dir)

        base_url = str(request.base_url).rstrip("/")
        if "localhost" not in base_url and "127.0.0.1" not in base_url:
            base_url = base_url.replace("http://", "https://")
        api_payload = _build_api_response(result, base_url, job_id)
        api_payload["pdf_status"] = "generating"
        _save_result_metadata(result_dir, api_payload)
        return PredictionResult(**api_payload)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@app.get("/results/{job_id}/files/{filename}")
async def get_result_file(job_id: str, filename: str) -> FileResponse:
    """Serve a saved pipeline image (original, segmentation, panels, etc.)."""
    safe_id = job_id.replace("..", "").replace("/", "")
    safe_name = Path(filename).name
    allowed = set(RESULT_IMAGE_FILES.values()) | {
        "optic_disc.png", "optic_cup.png", "resnet_result_image.png"
    }
    if safe_name not in allowed:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = RESULTS_DIR / safe_id / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/png")


@app.get("/results/{job_id}", response_model=PredictionResult)
async def get_result(job_id: str, request: Request) -> PredictionResult:
    """Load a previous analysis with all pipeline images from disk."""
    base_url = str(request.base_url).rstrip("/")
    if "localhost" not in base_url and "127.0.0.1" not in base_url:
        base_url = base_url.replace("http://", "https://")
    return PredictionResult(**_load_result_from_disk(job_id, base_url))


@app.get("/training/{filename}")
async def get_training_plot(filename: str) -> FileResponse:
    """Serve ResNet-50 training history plots."""
    safe = Path(filename).name
    if safe not in TRAINING_PLOTS:
        raise HTTPException(status_code=404, detail="Training plot not found")
    plot_path = TRAINING_DIR / safe
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Training plot file missing on server")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/eval/{filename}")
async def get_eval_plot(filename: str) -> FileResponse:
    """Serve model evaluation plots (ROC, confusion matrix, sample pipeline outputs)."""
    safe = Path(filename).name
    if safe not in EVAL_PLOTS:
        raise HTTPException(status_code=404, detail="Plot not found")
    plot_path = PLOTS_DIR / safe
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot file missing on server")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/reports/{report_id}.pdf")
async def download_report(report_id: str) -> FileResponse:
    safe_id = report_id.replace("..", "").replace("/", "")
    pdf_path = REPORTS_DIR / f"{safe_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{safe_id}.pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=False, log_level="info")
