"""Integration tests for the canonical FastAPI backend."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_fundus_bytes() -> bytes:
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 160, (40, 80, 40), -1)
    cv2.circle(img, (200, 200), 50, (180, 180, 180), -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "model_exists" in data
    assert data["version"] == "2.1.0"


def test_predict_missing_file(client):
    res = client.post("/predict")
    assert res.status_code == 422


@pytest.mark.skipif(
    not (PROJECT_ROOT / "outputs" / "models" / "best_model.pth").exists(),
    reason="Model weights not found",
)
def test_predict_full_workflow(client, sample_fundus_bytes):
    res = client.post(
        "/predict",
        files={"file": ("fundus.png", io.BytesIO(sample_fundus_bytes), "image/png")},
        data={"patient_name": "Test", "patient_age": "40", "patient_id": "T1"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["prediction"] in ("Glaucoma", "Normal")
    assert 0 <= data["confidence_score"] <= 100
    assert data["risk_level"] in ("Low", "Medium", "High")
    assert data["pdf_url"]
    assert data["preprocessing_image"] is None
    assert data["segmentation_panel_image"] is None
    assert data["cdr_report_image"] is None
    assert data["final_composite_image"] is None
    assert data["resnet_result_image"] is None
    assert data["cnn_prediction"] in ("GLAUCOMA", "NORMAL")
    assert data["segmentation_images"]["optic_disc"]

    report_id = data["report_id"]
    job_id = data["job_id"]
    pdf = None
    for _ in range(30):
        pdf = client.get(f"/reports/{report_id}.pdf")
        if pdf.status_code == 200:
            break
        time.sleep(0.5)
    assert pdf is not None and pdf.status_code == 200
    assert pdf.content[:4] == b"%PDF"

    reloaded = client.get(f"/results/{job_id}")
    assert reloaded.status_code == 200, reloaded.text
    reloaded_data = reloaded.json()
    assert reloaded_data["preprocessing_image"]
    assert reloaded_data["final_composite_image"]
    assert reloaded_data["prediction"] == data["prediction"]
