"""Server-side medical PDF report generation (optimized for speed)."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from config import OUTPUTS_DIR

PLOTS_DIR = OUTPUTS_DIR / "plots"
PRIMARY = colors.HexColor("#0A2540")  # Dark Navy
ACCENT = colors.HexColor("#00C2FF")   # Medical Cyan
TEXT = colors.HexColor("#0F172A")     # Off-Black Body Text
MUTED = colors.HexColor("#57687C")     # Slate Gray
BORDER = colors.HexColor("#E2E8F0")
GREEN = colors.HexColor("#10B981")    # Success Green
RED = colors.HexColor("#EF4444")      # Warning Red
AMBER = colors.HexColor("#F59E0B")    # Amber Alert
LIGHT_BG = colors.HexColor("#F8FAFC")  # Light background card
NAVY_BG = colors.HexColor("#0A2540")

_PDF_MAX_WIDTH_PX = 900
_JPEG_QUALITY = 72

DISK_FILES: dict[str, str] = {
    "resnet_result_image": "resnet_result_image.png",
    "preprocessing_image": "preprocessing_image.png",
    "segmentation_panel_image": "segmentation_panel_image.png",
    "cdr_report_image": "cdr_report_image.png",
    "pipeline_summary_image": "pipeline_summary_image.png",
    "final_composite_image": "final_composite_image.png",
}


def _diagnosis_color(prediction: str) -> colors.Color:
    p = prediction.lower()
    if "normal" in p and "borderline" not in p:
        return GREEN
    if "borderline" in p:
        return AMBER
    return RED


def _compress_image_bytes(raw: bytes) -> io.BytesIO:
    """Resize and JPEG-compress for fast PDF embedding."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if w > _PDF_MAX_WIDTH_PX:
        ratio = _PDF_MAX_WIDTH_PX / w
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.BOX)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    buf.seek(0)
    return buf


def _image_source(images: dict[str, Any], key: str, result_dir: Path | None) -> io.BytesIO | None:
    """Load image from disk (fast) or base64 fallback."""
    if result_dir:
        fname = DISK_FILES.get(key)
        if fname:
            path = result_dir / fname
            if path.exists():
                return _compress_image_bytes(path.read_bytes())
    b64 = images.get(key)
    if b64 and isinstance(b64, str) and not b64.startswith("http"):
        try:
            return _compress_image_bytes(base64.b64decode(b64))
        except Exception:
            return None
    return None


def _load_eval_image(name: str) -> io.BytesIO | None:
    path = PLOTS_DIR / name
    if path.exists():
        try:
            return _compress_image_bytes(path.read_bytes())
        except Exception:
            return None
    return None


def _rl_image(buf: io.BytesIO | None, width: float, height: float) -> RLImage | Paragraph:
    if buf is None:
        return Paragraph("<font color='#EF4444'><i>Clinical panel visualization unavailable.</i></font>", getSampleStyleSheet()["Normal"])
    return RLImage(buf, width=width, height=height)


def generate_medical_pdf(
    result: dict[str, Any],
    output_path: Path,
    result_dir: Path | None = None,
) -> Path:
    """Generate professional medical report structured into exactly 7 pages."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    
    # Document Style Elements
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Heading1"], fontSize=18, textColor=PRIMARY, spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        "DocSub", parent=styles["Heading3"], fontSize=9, textColor=MUTED, spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "PageTitle", parent=styles["Heading1"], fontSize=15, textColor=PRIMARY, spaceBefore=5, spaceAfter=10
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=10, textColor=PRIMARY, spaceBefore=12, spaceAfter=6
    )
    body = ParagraphStyle("BodyText", parent=styles["Normal"], fontSize=9, textColor=TEXT, leading=13)
    body_bold = ParagraphStyle("BodyTextBold", parent=body, fontName="Helvetica-Bold")
    caption_style = ParagraphStyle("CaptionText", parent=body, fontSize=8, textColor=MUTED, leading=11)
    
    disclaimer_style = ParagraphStyle(
        "DisclaimerText",
        parent=body,
        fontSize=8,
        textColor=MUTED,
        backColor=LIGHT_BG,
        borderColor=BORDER,
        borderWidth=0.5,
        borderPadding=10,
        leading=11,
    )

    prediction = result.get("prediction", "Unknown")
    diag_color = _diagnosis_color(prediction)
    patient = result.get("patient", {})
    report_id = result.get("report_id", "N/A")
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    images = result.get("images", {})
    cnn_pred = result.get("cnn_prediction", "—")
    resnet_prob = result.get("resnet_probability", "—")

    p_lower = prediction.lower()
    is_normal = "normal" in p_lower and "borderline" not in p_lower
    status_label = "NO GLAUCOMA INDICATORS DETECTED" if is_normal else "GLAUCOMA INDICATORS DETECTED"
    if "borderline" in p_lower:
        status_label = "BORDERLINE INDICATORS DETECTED"

    page_w = 180 * mm
    story: list[Any] = []

    # =========================================================================
    # PAGE 1: EXECUTIVE DIAGNOSTIC SUMMARY
    # =========================================================================
    story.append(Paragraph("Glaucoma Automated Screening Platform", title_style))
    story.append(Paragraph(f"Autonomous Clinical Assessment Report  &nbsp;|&nbsp;  System ID: {report_id}", subtitle_style))
    story.append(Spacer(1, 10))

    # Diagnosis Block
    story.append(Paragraph("PRIMARY CLINICAL IMPRESSION", section_style))
    story.append(
        Table(
            [[
                Paragraph(
                    f"<b><font size='13' color='{diag_color.hexval()}'>{status_label}</font></b><br/>"
                    f"<font color='{MUTED.hexval()}'>Fused Prediction Model consisting of ResNet-50 CNN Classifications "
                    f"and rule-based Cup-to-Disc Ratio (CDR) excavation analysis.</font>",
                    body
                )
            ]],
            colWidths=[page_w],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 1, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ])
        )
    )
    story.append(Spacer(1, 15))

    # Patient Details Card
    story.append(Paragraph("PATIENT DEMOGRAPHICS", section_style))
    story.append(
        Table(
            [
                [
                    Paragraph("<b>Patient Name:</b>", body),
                    Paragraph(patient.get("name", "Anonymous Patient"), body),
                    Paragraph("<b>Screening Date:</b>", body),
                    Paragraph(now, body),
                ],
                [
                    Paragraph("<b>Patient ID (MRN):</b>", body),
                    Paragraph(patient.get("id", "GLC-MRN-TEMP"), body),
                    Paragraph("<b>Date of Birth / Age:</b>", body),
                    Paragraph(f"{patient.get('age', 'Not Stated')} Years" if patient.get("age") else "Not Stated", body),
                ]
            ],
            colWidths=[35 * mm, 55 * mm, 35 * mm, 55 * mm],
            style=TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_BG]),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
    )
    story.append(Spacer(1, 15))

    # Clinical Scorecard Grid
    story.append(Paragraph("SCREENING METRICS SUMMARY", section_style))
    story.append(
        Table(
            [
                [Paragraph("<b>Clinical Measurement</b>", body_bold), Paragraph("<b>Observed Metric</b>", body_bold), Paragraph("<b>Referability Indicator</b>", body_bold)],
                [Paragraph("Fused Diagnosis Result", body), Paragraph(prediction, body_bold), Paragraph("Clinical Impression Output", body)],
                [Paragraph("ResNet-50 CNN Prediction", body), Paragraph(cnn_pred, body), Paragraph(f"Calibrated probability: {resnet_prob}", body)],
                [Paragraph("Cup-to-Disc Ratio (CDR)", body), Paragraph(f"{result.get('cup_disc_ratio', 0):.4f}", body_bold), Paragraph("Normal range is < 0.5; Excavated Suspected >= 0.6", body)],
                [Paragraph("Clinical Risk Assessment", body), Paragraph(result.get("risk_level", "—"), body_bold), Paragraph("Combined classifier confidence factor", body)],
                [Paragraph("Optic Disc Area", body), Paragraph(f"{result.get('disc_area', 0):,} px", body), Paragraph("Total structural disc canvas pixels", body)],
                [Paragraph("Optic Cup Area", body), Paragraph(f"{result.get('cup_area', 0):,} px", body), Paragraph("Total excavated cup cavity pixels", body)],
            ],
            colWidths=[55 * mm, 45 * mm, 80 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: PREPROCESSING PIPELINE
    # =========================================================================
    story.append(Paragraph("STAGE 1: IMAGE PREPROCESSING PIPELINE", h1_style))
    story.append(Paragraph(
        "A critical phase in automated retinal screening is the normalization of RGB fundus photography to reduce visual noise, "
        "neutralize varying lighting conditions, and enhance boundary structures for optic disc localization.",
        body
    ))
    story.append(Spacer(1, 15))

    buf_preprocess = _image_source(images, "preprocessing_image", result_dir)
    story.append(_rl_image(buf_preprocess, page_w, 90 * mm))
    story.append(Spacer(1, 15))

    story.append(Paragraph("PIPELINE STAGE INTERPRETATION & METHODOLOGY", section_style))
    story.append(
        Table(
            [
                [Paragraph("<b>Step</b>", body_bold), Paragraph("<b>Algorithmic Method</b>", body_bold), Paragraph("<b>Clinical Utility</b>", body_bold)],
                [
                    Paragraph("Green Channel Extraction", body),
                    Paragraph("Isolates the green sensor channel from the RGB source.", body),
                    Paragraph("Offers the highest contrast for blood vessels and optic cup boundaries.", body)
                ],
                [
                    Paragraph("CLAHE Enhancement", body),
                    Paragraph("Applies Contrast Limited Adaptive Histogram Equalization.", body),
                    Paragraph("Amplifies local texture detail without over-saturating illumination peaks.", body)
                ],
                [
                    Paragraph("Gaussian Noise Suppression", body),
                    Paragraph("Applies gaussian blur convolution filter.", body),
                    Paragraph("Suppresses high-frequency noise and sensory artifact interference.", body)
                ]
            ],
            colWidths=[40 * mm, 70 * mm, 70 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: ENHANCED K-STRANGE SEGMENTATION
    # =========================================================================
    story.append(Paragraph("STAGE 2: DETERMINISTIC K-STRANGE SEGMENTATION", h1_style))
    story.append(Paragraph(
        "Following preprocessing, the system segments the optic nerve head. Unlike black-box neural networks, "
        "our system employs the Enhanced K-Strange Points clustering algorithm to produce reproducible, geometric "
        "boundaries of the optic disc and cup.",
        body
    ))
    story.append(Spacer(1, 15))

    buf_seg = _image_source(images, "segmentation_panel_image", result_dir)
    story.append(_rl_image(buf_seg, page_w, 95 * mm))
    story.append(Spacer(1, 15))

    story.append(Paragraph("SEGMENTATION PRINCIPLES & GEOMETRIC ANCHORS", section_style))
    story.append(Paragraph(
        "<b>Enhanced K-Strange Points Clustering:</b> The algorithm operates in two passes. First, it isolates the optic disc ROI "
        "from surrounding retinal background using Kmin and Kmax anchors mapped to local intensity minima and maxima. Second, it shifts the "
        "coordinate system within the segmented disc to isolate the excavated optic cup based on pixel illumination differences. "
        "This deterministic approach ensures boundaries are physically justifiable and auditable by clinicians.",
        body
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: RESNET-50 CLASSIFICATION
    # =========================================================================
    story.append(Paragraph("STAGE 3: DEEP LEARNING CLASSIFICATION", h1_style))
    story.append(Paragraph(
        "Complementary to the geometric segmentation, the system runs the cropped optic disc ROI through a deep learning "
        "classification network. Fine-tuned on a corpus of 9,005 fundus photographs, the model evaluates global shape, "
        "rim erosion, and tissue texture.",
        body
    ))
    story.append(Spacer(1, 15))

    buf_resnet = _image_source(images, "resnet_result_image", result_dir)
    story.append(_rl_image(buf_resnet, page_w, 95 * mm))
    story.append(Spacer(1, 15))

    story.append(Paragraph("RESNET-50 ARCHITECTURE & NEURAL ANALYSIS", section_style))
    story.append(Paragraph(
        f"<b>Sigmoid Classification Score:</b> The neural net outputs a calibrated posterior probability of <b>{resnet_prob}</b>, "
        f"yielding a model prediction label of <b>{cnn_pred}</b>. The fine-tuned ResNet-50 uses focal loss optimizations during training "
        f"to specialize in early-stage glaucoma classification features that might escape standard geometric rules. Fusing the deep learning "
        f"score with segmented CDR diameters provides high specificity and sensitivity.",
        body
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: INTERPRETATION & RISK ASSESSMENT
    # =========================================================================
    story.append(Paragraph("CLINICAL RISK ASSESSMENT & DIALECTIC ANALYSIS", h1_style))
    story.append(Paragraph(
        "A hybrid approach resolves disagreements between computer vision segmentation and deep learning classification. "
        "The diagnostic decision flow maps geometric cup excavation along with CNN features.",
        body
    ))
    story.append(Spacer(1, 15))

    buf_cdr = _image_source(images, "cdr_report_image", result_dir)
    story.append(_rl_image(buf_cdr, page_w, 95 * mm))
    story.append(Spacer(1, 15))

    story.append(Paragraph("CDR GEOMETRIC CLASSIFICATION REFERENCE", section_style))
    story.append(
        Table(
            [
                [Paragraph("<b>Cup-to-Disc Ratio (CDR)</b>", body_bold), Paragraph("<b>Clinical Interpretation & Directive</b>", body_bold)],
                [Paragraph("&lt; 0.5", body), Paragraph("Physiological norm. Low likelihood of excavation damage.", body)],
                [Paragraph("0.5 – 0.6", body), Paragraph("Borderline. Close monitoring and tracking of visual field trends recommended.", body)],
                [Paragraph("0.6 – 0.8", body), Paragraph("Glaucoma Suspected. Clear indicator of optic nerve excavation. Specialist referral recommended.", body)],
                [Paragraph("&gt; 0.8", body), Paragraph("Severe Excavation. Urgent referral to glaucoma specialist required.", body)]
            ],
            colWidths=[55 * mm, 125 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: MODEL PERFORMANCE & ROC
    # =========================================================================
    story.append(Paragraph("MODEL PERFORMANCE & CLINICAL VALIDATION", h1_style))
    story.append(Paragraph(
        "To ensure clinical-grade screening, the ResNet-50 + K-Strange pipeline was evaluated on a held-out test set (n=1,353 images) "
        "drawn from multi-ethnic benchmarks (REFUGE, RIM-ONE, DRISHTI-GS, G1020).",
        body
    ))
    story.append(Spacer(1, 15))

    buf_roc = _load_eval_image("roc_curve.png")
    story.append(_rl_image(buf_roc, page_w, 95 * mm))
    story.append(Spacer(1, 15))

    # Balanced Model Metrics Table
    story.append(Paragraph("BALANCED CLASSIFICATION SCORECARD", section_style))
    story.append(
        Table(
            [
                [Paragraph("<b>Evaluation Metric</b>", body_bold), Paragraph("<b>Value</b>", body_bold), Paragraph("<b>Clinical Significance</b>", body_bold)],
                [Paragraph("Test Accuracy", body), Paragraph("89.28%", body_bold), Paragraph("Overall correct diagnosis rate across balanced cohorts.", body)],
                [Paragraph("Sensitivity (Recall)", body), Paragraph("83.07%", body_bold), Paragraph("Catches true positive cases to prevent dangerous false negatives.", body)],
                [Paragraph("Specificity", body), Paragraph("93.02%", body_bold), Paragraph("Minimizes false alarms and unnecessary specialist referrals.", body)],
                [Paragraph("Model Precision", body), Paragraph("87.73%", body_bold), Paragraph("Positive predictive value of positive screens.", body)],
                [Paragraph("Area Under ROC (AUC)", body), Paragraph("0.961", body_bold), Paragraph("Superb discriminatory capacity across all classification thresholds.", body)],
            ],
            colWidths=[50 * mm, 30 * mm, 100 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BG),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("PADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: RECOMMENDATIONS & DISCLAIMERS
    # =========================================================================
    story.append(Paragraph("DIAGNOSTIC RECOMMENDATIONS & DISCLAIMER", h1_style))
    story.append(Paragraph(
        "Based on the observed metrics, the following clinical recommendations and directives are suggested. "
        "These guidelines conform to standard screening protocols.",
        body
    ))
    story.append(Spacer(1, 15))

    story.append(Paragraph("RECOMMENDED FOLLOW-UP DIRECTIVES", section_style))
    for i, rec in enumerate(result.get("recommendations", [])):
        story.append(Paragraph(f"<b>{i + 1}.</b> {rec}", body))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 15))
    story.append(Paragraph("CLINICIAN SIGN-OFF BLOCK", section_style))
    story.append(
        Table(
            [
                [Paragraph("<b>Examining Ophthalmologist:</b>", body), Paragraph("___________________________", body)],
                [Paragraph("<b>Signature:</b>", body), Paragraph("___________________________", body)],
                [Paragraph("<b>Sign-Off Date:</b>", body), Paragraph(datetime.now().strftime("%d %b %Y"), body)]
            ],
            colWidths=[60 * mm, 120 * mm],
            style=TableStyle([
                ("LINEBELOW", (1, 0), (1, 1), 0.5, colors.gray),
                ("PADDING", (0, 0), (-1, -1), 10),
            ])
        )
    )
    story.append(Spacer(1, 25))

    # Disclaimer Card
    disclaimer_text = (
        "<b>MEDICAL SCREENING DISCLAIMER:</b> This report is generated automatically by an automated deep learning screening platform. "
        "It is designed solely to assist clinicians as an auxiliary diagnostic tool. All findings, including segmented boundaries, "
        "calculated Cup-to-Disc Ratios, and CNN probabilities, must be reviewed and validated by a licensed eye-care professional. "
        "This software does not make a final medical diagnosis or replace comprehensive clinical ophthalmic examinations. "
        "MITE ISE 2025–26."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return output_path

