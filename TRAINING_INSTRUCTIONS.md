# Glaucoma Detection - Enhanced Training Instructions

## Improvements Made for Better Accuracy

I've made the following enhancements to improve upon the baseline accuracy of 81.84%:

### 1. Enhanced Training Configuration (`src/train.py`)
- **Increased epochs**: 20 → 40 total epochs
- **Phase 1 (frozen backbone)**: 5 → 8 epochs
- **Phase 2 (fine-tuning)**: 15 → 32 epochs  
- **Learning rate phase 1**: 1e-3 → 5e-4 (more stable start)
- **Early stopping patience**: 6 → 10 (allows longer training)
- **Batch size**: 8 (maintained for CPU stability)
- **Dropout**: 0.5 (maintained)
- **Decision threshold**: 0.45 (recall-friendly, maintained)

### 2. Enhanced Data Augmentation (`src/dataset.py`)
- **Added RandomResizedCrop**: For scale and aspect ratio variation
- **Increased rotation**: 30° → 45° 
- **Enhanced ColorJitter**: 
  - Brightness: 0.2 → 0.3
  - Contrast: 0.2 → 0.3
  - Saturation: 0.1 → 0.2
  - Hue: 0.05 → 0.1
- **Added RandomGrayscale** (p=0.1): For illumination invariance
- **Added GaussianBlur**: To simulate defocus/motion blur
- **Kept essential transforms**: Horizontal/vertical flips, ToTensor, ImageNet normalization

## How to Run the Enhanced Training

### Step 1: Set up Virtual Environment
```bash
# Navigate to glaucoma_project directory
cd glaucoma_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 3: Start Training
```bash
# Start training (will show progress in terminal)
python src/train.py

# Alternatively, run in background and monitor log:
python src/train.py > training.log 2>&1 &
# Then monitor with: Get-Content training.log -Wait -Tail 20  (PowerShell)
# or: tail -f training.log  (Unix/macOS)
```

### Step 4: Monitor Progress
You should see output similar to:
```
============================================================
GLAUCOMA DETECTION — TRAINING PIPELINE
============================================================
Device: CPU
Total epochs: 40
Batch size: 8
Decision threshold: 0.45
============================================================
Loss: using FocalLoss(alpha=X.XXXX, gamma=2.0)
------------------------------------------------------------
PHASE 1: Training classification head only (Epochs 1-8)
------------------------------------------------------------
...
[PHASE 1] Epoch 1/8
  Learning rate: 0.000500
    Batch [10/788] Loss: 0.XXXX
    Batch [20/788] Loss: 0.XXXX
    ...
Epoch 1 Summary — Train Loss: X.XXXX | Train Acc: X.XX% | Val Loss: X.XXXX | Val Acc: X.XX% | Time: X.Xs
...
```

### Step 5: Evaluate Results
After training completes, evaluate with threshold tuning:
```bash
python src/evaluate.py
```
This will show:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Optimal threshold for recall-focused tuning
- Confusion matrix
- Detailed metrics

## Expected Improvements
With these enhancements, you can expect:
- **Better generalization** from enhanced augmentation
- **More stable convergence** from adjusted learning rates
- **Sufficient training time** from increased epochs and patience
- **Potential accuracy improvement** from 81.84% toward 85-90%+ range
- **Better recall** (critical for glaucoma screening) maintained or improved

## Notes
- Training time will increase due to more epochs (approximately 2x longer)
- The model will save the best version to `outputs/models/best_model.pth`
- Training logs and plots will be saved in the `outputs/` directory
- If you encounter CUDA Out of Memory errors and have a GPU, consider reducing batch size
- For CPU-only training (as seen in logs), the current settings are optimized for stability

Let the training run to completion and then evaluate to see the improvement over the baseline 81.84% accuracy!