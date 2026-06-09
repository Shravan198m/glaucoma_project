# Glaucoma Detection System - Setup & User Guide

## 🚀 Installation & Setup

### Prerequisites
- **Python:** 3.9 or later
- **Operating System:** Windows, macOS, or Linux
- **Memory:** Minimum 8GB RAM (16GB+ recommended)
- **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for acceleration

### Step 1: Environment Setup

```bash
# Clone/download the project
cd glaucoma_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install all dependencies with specific versions
pip install -r requirements.txt
```

### Optional: Generate a pinned lockfile

After you've created and validated your virtual environment, generate a lockfile to ensure reproducible installs:

```bash
pip freeze > requirements-lock.txt
```

Commit `requirements-lock.txt` and update CI to use it for deterministic installs.

### Step 3: Verify Installation

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verify CUDA availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test configuration module
python src/config.py
```

---

## 📋 Usage Guide

### Option 1: Single Image Processing

#### Preprocessing Only
```bash
cd glaucoma_project
python src/preprocessing.py
```
**Output:** 
- Console visualization of preprocessing stages
- Saved to: `outputs/preprocessed/`

#### Full Pipeline (Recommended)
```bash
python src/cdr.py
```
**Includes:**
1. Image preprocessing (green channel, CLAHE, filtering)
2. ROI extraction and normalization
3. Disc/cup segmentation (K-Strange clustering)
4. CDR calculation with clinical interpretation
5. CNN inference (if model available)
6. Decision fusion and final diagnosis

**Output:**
- Clinical report with visualizations
- Saved to: `outputs/results/`

#### CNN Inference Only
```bash
python src/inference.py
```
**Output:** 
- Probability score (0-1)
- Classification (Normal/Glaucoma)

---

### Option 2: Batch Processing (9,005 Test Images)

```bash
# Process all images in batch mode
python src/pipeline.py batch

# Wait for completion (may take 30-60 minutes on CPU)

# Aggregate results and compute metrics
python src/aggregate_results.py
```

**Output:**
- Individual results: `outputs/results/`
- Metrics summary: Console and file
- Visualizations: `outputs/plots/`

---

### Option 3: Model Training

#### Quick Training (20 epochs, unfrozen for demo)
```bash
python src/train.py
```

#### Optimized Training (50 epochs, best practices)
```bash
python src/train_optimized.py
```

**Configuration:**
- Modify `src/config.py` to adjust:
  - Batch size
  - Learning rates
  - Number of epochs
  - Early stopping patience
  - Device (CPU vs CUDA)

**Output:**
- Trained model: `outputs/models/best_model.pth`
- Training plots: `outputs/plots/`
- Metrics: Console output

---

### Option 4: Evaluation on Test Set

```bash
python src/evaluate.py
```

**Output:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve
- Classification Report

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/test_suite.py -v
```

### Run Specific Test
```bash
pytest tests/test_suite.py::test_config_initialization -v
```

### Run with Coverage Report
```bash
pip install pytest-cov
pytest tests/test_suite.py --cov=src --cov-report=html
```

---

## ⚙️ Configuration Management

### View Current Configuration
```bash
python src/config.py
```

### Modify Configuration

Edit `src/config.py` to change:

**Preprocessing:**
```python
preprocessing_config = PreprocessingConfig(
    kernel_size=5,
    gaussian_sigma=1.0,
    clahe_clip_limit=2.0,
    use_clahe=True,
)
```

**Training:**
```python
training_config = TrainingConfig(
    batch_size=16,
    num_epochs=50,
    lr_phase1=5e-4,
    lr_phase2=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
```

**Segmentation:**
```python
segmentation_config = SegmentationConfig(
    roi_size=(200, 200),
    cup_threshold_percentile=65.0,
)
```

---

## 📊 Understanding the Output

### Classification Report
```
              precision    recall  f1-score   support
       Normal       0.89      0.89      0.89      5019
    Glaucoma       0.80      0.80      0.80      3985
    Weighted       0.86      0.86      0.86      9004
```

**What Each Metric Means:**
- **Precision:** Of predicted glaucoma cases, how many are correct?
- **Recall:** Of all actual glaucoma cases, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of samples in each class

### Confusion Matrix
```
         Predicted
       Normal  Glaucoma
Actual
Normal    TN      FP
Glaucoma  FN      TP
```

- **TP (True Positive):** Correctly identified glaucoma ✅
- **TN (True Negative):** Correctly identified normal ✅
- **FP (False Positive):** Incorrectly marked as glaucoma (false alarm) ⚠️
- **FN (False Negative):** Missed glaucoma (DANGEROUS!) ❌

### CDR (Cup-to-Disc Ratio) Interpretation

| CDR Value | Status | Risk Level | Recommendation |
|-----------|--------|-----------|-----------------|
| < 0.5 | Normal | Low | Annual checkup |
| 0.5-0.6 | Borderline | Moderate | Follow-up in 6 months |
| 0.6-0.8 | Glaucoma Suspected | High | Specialist consultation |
| > 0.8 | Advanced Glaucoma | Very High | Urgent specialist visit |

---

## 🔧 Troubleshooting

### Issue: CUDA not available
```
Solution: Update NVIDIA drivers or reinstall PyTorch CPU version
```

### Issue: Out of memory (OOM)
```
Solution: 
1. Reduce batch_size in src/config.py
2. Close other applications
3. Use CPU instead of GPU
```

### Issue: Module not found error
```
Solution:
cd glaucoma_project
python -m pip install -r requirements.txt --force-reinstall
```

### Issue: Image processing fails
```
Solution:
- Ensure image is valid JPG/PNG
- Check image resolution (should be 400x400+ pixels)
- Verify image is not corrupted
```

---

## 📈 Performance Tuning

### For Speed (GPU):
```python
# In src/config.py
training_config.batch_size = 32  # Larger batches
training_config.num_workers = 4   # Parallel loading
```

### For Accuracy:
```python
# In src/config.py  
training_config.num_epochs = 100  # More training
training_config.early_stop_patience = 20  # Allow longer plateaus
```

### For Memory Efficiency:
```python
# In src/config.py
training_config.batch_size = 8   # Smaller batches
training_config.device = 'cpu'    # Use CPU
```

---

## 📁 Project Structure

```
glaucoma_project/
├── src/
│   ├── config.py              # Centralized configuration
│   ├── error_handler.py       # Error handling & logging
│   ├── preprocessing.py       # Image preprocessing
│   ├── segmentation.py        # Disc/cup segmentation
│   ├── cdr.py                 # CDR calculation
│   ├── model.py               # ResNet-50 model
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation & metrics
│   ├── pipeline.py            # Full end-to-end pipeline
│   ├── dataset.py             # Dataset utilities
│   └── aggregate_results.py   # Result aggregation
├── tests/
│   └── test_suite.py          # Comprehensive tests
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_segmentation.ipynb
│   ├── ...
│   └── 06_final_output.ipynb
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── outputs/
│   ├── models/                # Trained models
│   ├── results/               # Inference results
│   ├── plots/                 # Visualizations
│   └── logs/                  # Log files
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── SETUP_GUIDE.md            # This file
```

---

## 🔐 Best Practices

1. **Always use virtual environment** to avoid package conflicts
2. **Keep requirements.txt updated** when adding dependencies
3. **Set random seed** for reproducible results
4. **Validate input images** before processing
5. **Monitor GPU memory** when training
6. **Save checkpoints** during training
7. **Log all operations** for debugging
8. **Test on small dataset first** before batch processing

---

## 🤝 Contributing

To improve the project:
1. Add new tests in `tests/test_suite.py`
2. Update configuration in `src/config.py`
3. Document changes in comments
4. Run full test suite before committing
5. Update this guide if adding new features

---

## 📞 Support & Issues

For issues or questions:
1. Check troubleshooting section above
2. Review error messages carefully
3. Check `outputs/logs/` for detailed logs
4. Verify all dependencies installed correctly
5. Test with sample images first

---

**Last Updated:** May 3, 2026  
**Status:** ✅ Production Ready with Enhanced Error Handling
