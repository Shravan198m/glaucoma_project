# Glaucoma Detection API — Instructions

## Architecture

The **canonical backend** is FastAPI in `src/api.py` (port **8000**).

```
Frontend (glaucoma-app) → FastAPI (src/api.py) → prediction_service.py → AI modules
                                                      ↓
                                               pdf_service.py → /reports/{id}.pdf
```

There is no separate Flask server. All inference uses `glaucoma_project/src/`.

---

## Prerequisites

1. Python 3.9+
2. Model weights at `outputs/models/best_model.pth`
3. Dependencies: `pip install -r requirements.txt`

---

## Start the API

### Option 1 — Scripts (recommended)

**Windows:** `run_api.bat`  
**PowerShell:** `run_api.ps1`

### Option 2 — Manual

```bash
cd glaucoma_project
pip install -r requirements.txt
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Start the frontend

```bash
cd glaucoma-app
npm install
npm run dev
```

Open http://localhost:5173 — Vite proxies API calls to port 8000.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server and model status |
| POST | `/predict` | Upload fundus image, run full pipeline |
| GET | `/reports/{report_id}.pdf` | Download generated PDF report |
| GET | `/docs` | Swagger UI |

---

## POST /predict

**Content-Type:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `file` | Yes | Fundus image (JPG, PNG, BMP, TIFF) |
| `patient_name` | No | Patient name for PDF |
| `patient_age` | No | Patient age |
| `patient_id` | No | Patient ID |

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@fundus.jpg" \
  -F "patient_name=John Doe" \
  -F "patient_age=45" \
  -F "patient_id=P001"
```

**Response:**

```json
{
  "report_id": "GLC-A1B2C3D4",
  "prediction": "Normal",
  "confidence_score": 87.5,
  "cup_disc_ratio": 0.4521,
  "risk_level": "Low",
  "segmentation_images": {
    "original": "<base64>",
    "optic_disc": "<base64>",
    "optic_cup": "<base64>",
    "segmentation": "<base64>",
    "vessels": "<base64>"
  },
  "heatmap_image": "<base64>",
  "recommendations": ["..."],
  "pdf_url": "http://localhost:8000/reports/GLC-A1B2C3D4.pdf"
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API port |
| `GLAUCOMA_DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

---

## Module Map (single source of truth)

| Purpose | Module |
|---------|--------|
| API routes | `src/api.py` |
| Pipeline orchestration | `src/prediction_service.py` |
| PDF reports | `src/pdf_service.py` |
| Preprocessing | `src/preprocessing.py` |
| K-Strange segmentation | `src/segmentation.py` |
| CDR | `src/cdr.py` |
| ResNet-50 | `src/model.py` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not loaded | Verify `outputs/models/best_model.pth` exists |
| 422 on /predict | Use form field `file`, not `image` |
| CORS errors | Start API with CORS enabled; use Vite proxy in dev |
| PDF 404 | PDF is created per prediction; use `pdf_url` from response |
