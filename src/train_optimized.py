"""
OPTIMIZED TRAINING PIPELINE FOR GLAUCOMA DETECTION
Implements advanced techniques for maximum accuracy
"""

import copy
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from dataset import create_dataloaders
from model import create_resnet50_model, freeze_all_except_head, unfreeze_layer4_and_head


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZED TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_OPTIMIZED = {
    'dataset_root': str(PROJECT_ROOT / 'dataset'),
    'batch_size': 16,                    # Increased from 8 for stable gradients
    'num_epochs': 50,                    # Increased from 20 for better convergence
    'phase1_epochs': 8,                  # Increased from 5
    'phase2_epochs': 42,                 # Increased fine-tuning epochs
    'lr_phase1': 5e-4,                   # Reduced for stability
    'lr_phase2': 1e-4,                   # Fine-tuning LR
    'dropout': 0.4,                      # Reduced from 0.5
    'decision_threshold': 0.45,          # Recall-friendly threshold
    'early_stop_patience': 12,           # Increased patience
    'early_stop_min_delta': 0.0005,      # Stricter improvement threshold
    'save_path': str(PROJECT_ROOT / 'outputs' / 'models' / 'best_model_optimized.pth'),
    'device': 'cpu',
    'use_lr_scheduler': True,            # Enable learning rate scheduling
    'gradient_clip': 1.0,                # Prevent exploding gradients
    'weight_decay': 1e-4,                # L2 regularization
}

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED LOSS FUNCTION WITH FOCAL LOSS OPTION
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance more effectively."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, outputs, labels):
        """
        Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        Focuses training on hard examples
        """
        outputs = torch.clamp(outputs, 1e-6, 1 - 1e-6)
        labels = labels.float()

        pt = torch.where(labels >= 0.5, outputs, 1.0 - outputs)
        alpha_t = torch.where(
            labels >= 0.5,
            torch.as_tensor(self.alpha, device=outputs.device, dtype=outputs.dtype),
            torch.as_tensor(1.0 - self.alpha, device=outputs.device, dtype=outputs.dtype),
        )

        focal_weight = (1.0 - pt) ** self.gamma
        focal_loss = -alpha_t * focal_weight * torch.log(pt)
        return focal_loss.mean()


def predict_from_outputs(outputs, threshold):
    """Convert probabilities to binary predictions using a threshold."""
    return (outputs >= threshold).float()


def get_loss_function(class_weights, use_focal=True):
    """
    Enhanced loss function with optional focal loss.
    Focal loss gives more weight to hard examples.
    """
    class_weights = class_weights.float()
    normal_weight = class_weights[0].item()
    glaucoma_weight = class_weights[1].item()
    
    print(f"\n  Loss Configuration:")
    print(f"  ├─ Loss Type: {'Focal Loss' if use_focal else 'Weighted BCE'}")
    print(f"  ├─ Normal weight:   {normal_weight:.4f}")
    print(f"  └─ Glaucoma weight: {glaucoma_weight:.4f}")
    
    if use_focal:
        focal_loss = FocalLoss(alpha=normal_weight, gamma=2.0)
        
        def criterion(outputs, labels):
            # Focal loss already emphasizes difficult positive cases; keep the
            # additional class weighting lightweight and per-sample.
            sample_weights = torch.where(
                labels >= 0.5,
                torch.as_tensor(glaucoma_weight, device=outputs.device, dtype=outputs.dtype),
                torch.as_tensor(normal_weight, device=outputs.device, dtype=outputs.dtype),
            )
            return focal_loss(outputs, labels) * sample_weights.mean()
    else:
        def criterion(outputs, labels):
            sample_weights = torch.where(
                labels >= 0.5,
                torch.tensor(glaucoma_weight, device=outputs.device, dtype=outputs.dtype),
                torch.tensor(normal_weight, device=outputs.device, dtype=outputs.dtype),
            )
            return F.binary_cross_entropy(outputs, labels, weight=sample_weights)
    
    return criterion


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING WITH GRADIENT CLIPPING & MIXED PRECISION
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch_optimized(model, loader, criterion, optimizer, device, epoch, config, threshold):
    """Enhanced training epoch with gradient clipping and detailed logging."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        
        optimizer.step()
        
        running_loss += loss.item()
        predicted = predict_from_outputs(outputs, threshold)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            current_acc = 100.0 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f"      Batch [{batch_idx + 1:3d}/{batch_count}] Loss: {avg_loss:.4f} | Acc: {current_acc:.2f}%")
    
    epoch_loss = running_loss / batch_count
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate_one_epoch_optimized(model, loader, criterion, device, threshold):
    """Enhanced validation with confidence metrics."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = predict_from_outputs(outputs, threshold)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_probs.extend(outputs.squeeze().cpu().numpy().tolist())
    
    val_loss = running_loss / len(loader)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED EARLY STOPPING WITH CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStoppingOptimized:
    """Advanced early stopping with best model tracking."""
    
    def __init__(self, patience=12, min_delta=0.0005, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_model = None
        self.stop = False
    
    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.best_model, self.save_path)
            print(f"    💾 ★ BEST MODEL SAVED ★ Epoch {epoch} | Val Loss: {val_loss:.6f}")
            return True  # Model improved
        else:
            self.counter += 1
            remaining = self.patience - self.counter
            print(f"    ⏳ No improvement for {self.counter} epoch(s). Patience: {remaining}/{self.patience}")
            
            if self.counter >= self.patience:
                self.stop = True
                print(f"    🛑 EARLY STOPPING TRIGGERED! Best model was at epoch {self.best_epoch}")
            
            return False  # No improvement


# ─────────────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZED TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_model_optimized():
    """Optimized training pipeline with all advanced techniques."""
    
    print("\n" + "=" * 75)
    print("🔬 GLAUCOMA DETECTION SYSTEM — OPTIMIZED TRAINING PIPELINE 🔬")
    print("=" * 75)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)
    print("\n📊 CONFIGURATION:")
    print(f"  ├─ Device: {CONFIG_OPTIMIZED['device'].upper()}")
    print(f"  ├─ Batch Size: {CONFIG_OPTIMIZED['batch_size']}")
    print(f"  ├─ Total Epochs: {CONFIG_OPTIMIZED['num_epochs']}")
    print(f"  ├─ Phase 1 Epochs: {CONFIG_OPTIMIZED['phase1_epochs']} (frozen backbone)")
    print(f"  ├─ Phase 2 Epochs: {CONFIG_OPTIMIZED['phase2_epochs']} (fine-tuning)")
    print(f"  ├─ LR Phase 1: {CONFIG_OPTIMIZED['lr_phase1']}")
    print(f"  ├─ LR Phase 2: {CONFIG_OPTIMIZED['lr_phase2']}")
    print(f"  ├─ Dropout: {CONFIG_OPTIMIZED['dropout']}")
    print(f"  ├─ Decision Threshold: {CONFIG_OPTIMIZED['decision_threshold']}")
    print(f"  ├─ Gradient Clip: {CONFIG_OPTIMIZED['gradient_clip']}")
    print(f"  ├─ Weight Decay: {CONFIG_OPTIMIZED['weight_decay']}")
    print(f"  ├─ LR Scheduler: {'Enabled' if CONFIG_OPTIMIZED['use_lr_scheduler'] else 'Disabled'}")
    print(f"  └─ Early Stop Patience: {CONFIG_OPTIMIZED['early_stop_patience']} epochs")
    print("=" * 75)
    
    device = torch.device(CONFIG_OPTIMIZED['device'])
    
    # Load data
    print("\n📂 Loading dataset...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        dataset_root=CONFIG_OPTIMIZED['dataset_root'],
        batch_size=CONFIG_OPTIMIZED['batch_size'],
    )
    print(f"  ✅ Train: {len(train_loader) * CONFIG_OPTIMIZED['batch_size']} images")
    print(f"  ✅ Val: {len(val_loader) * CONFIG_OPTIMIZED['batch_size']} images")
    print(f"  ✅ Test: {len(test_loader) * CONFIG_OPTIMIZED['batch_size']} images")
    
    # Create model
    print("\n🧠 Building model...")
    model = create_resnet50_model(dropout_rate=CONFIG_OPTIMIZED['dropout'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Model Parameters: {total_params:,}")
    
    # Loss function with focal loss
    criterion = get_loss_function(class_weights, use_focal=True)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
        'epoch': [],
    }
    
    early_stopping = EarlyStoppingOptimized(
        patience=CONFIG_OPTIMIZED['early_stop_patience'],
        min_delta=CONFIG_OPTIMIZED['early_stop_min_delta'],
        save_path=CONFIG_OPTIMIZED['save_path']
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Train only head (frozen backbone)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("⚙️  PHASE 1: BACKBONE FROZEN (Head Training)")
    print("=" * 75)
    
    freeze_all_except_head(model)
    optimizer_phase1 = optim.Adam(
        model.parameters(),
        lr=CONFIG_OPTIMIZED['lr_phase1'],
        weight_decay=CONFIG_OPTIMIZED['weight_decay']
    )
    
    if CONFIG_OPTIMIZED['use_lr_scheduler']:
        scheduler_phase1 = ReduceLROnPlateau(
            optimizer_phase1, mode='min', factor=0.5, patience=3
        )
    
    for epoch in range(1, CONFIG_OPTIMIZED['phase1_epochs'] + 1):
        print(f"\n📍 Epoch {epoch}/{CONFIG_OPTIMIZED['phase1_epochs']} (Phase 1)")
        
        train_loss, train_acc = train_one_epoch_optimized(
            model,
            train_loader,
            criterion,
            optimizer_phase1,
            device,
            epoch,
            CONFIG_OPTIMIZED,
            CONFIG_OPTIMIZED['decision_threshold'],
        )
        val_loss, val_acc, _ = validate_one_epoch_optimized(
            model,
            val_loader,
            criterion,
            device,
            CONFIG_OPTIMIZED['decision_threshold'],
        )
        
        print(f"  📈 Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        print(f"  📊 Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer_phase1.param_groups[0]['lr'])
        history['epoch'].append(epoch)
        
        if CONFIG_OPTIMIZED['use_lr_scheduler']:
            scheduler_phase1.step(val_loss)
        
        improved = early_stopping(val_loss, model, epoch)
        
        if early_stopping.stop:
            print(f"\n🛑 Early stopping at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Fine-tune layer4 + head
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("⚙️  PHASE 2: LAYER4 + HEAD FINE-TUNING")
    print("=" * 75)
    
    # Reset early stopping for phase 2
    early_stopping.counter = 0
    early_stopping.patience = CONFIG_OPTIMIZED['early_stop_patience']
    
    unfreeze_layer4_and_head(model)
    optimizer_phase2 = optim.Adam(
        model.parameters(),
        lr=CONFIG_OPTIMIZED['lr_phase2'],
        weight_decay=CONFIG_OPTIMIZED['weight_decay']
    )
    
    if CONFIG_OPTIMIZED['use_lr_scheduler']:
        scheduler_phase2 = CosineAnnealingLR(
            optimizer_phase2,
            T_max=CONFIG_OPTIMIZED['phase2_epochs'],
            eta_min=1e-6
        )
    
    phase2_start = CONFIG_OPTIMIZED['phase1_epochs'] + 1
    phase2_end = CONFIG_OPTIMIZED['num_epochs'] + 1
    
    for epoch in range(phase2_start, phase2_end):
        phase2_epoch = epoch - CONFIG_OPTIMIZED['phase1_epochs']
        print(f"\n📍 Epoch {epoch}/{CONFIG_OPTIMIZED['num_epochs']} (Phase 2 - Epoch {phase2_epoch})")
        
        train_loss, train_acc = train_one_epoch_optimized(
            model,
            train_loader,
            criterion,
            optimizer_phase2,
            device,
            epoch,
            CONFIG_OPTIMIZED,
            CONFIG_OPTIMIZED['decision_threshold'],
        )
        val_loss, val_acc, _ = validate_one_epoch_optimized(
            model,
            val_loader,
            criterion,
            device,
            CONFIG_OPTIMIZED['decision_threshold'],
        )
        
        current_lr = optimizer_phase2.param_groups[0]['lr']
        print(f"  📈 Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.2f}%")
        print(f"  📊 Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        history['epoch'].append(epoch)
        
        if CONFIG_OPTIMIZED['use_lr_scheduler']:
            scheduler_phase2.step()
        
        improved = early_stopping(val_loss, model, epoch)
        
        if early_stopping.stop:
            print(f"\n🛑 Early stopping at epoch {epoch}. Best epoch: {early_stopping.best_epoch}")
            break
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOAD BEST MODEL & EVALUATE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("📋 LOADING BEST MODEL & FINAL EVALUATION")
    print("=" * 75)
    
    if early_stopping.best_model:
        model.load_state_dict(early_stopping.best_model)
        print(f"✅ Loaded best model from epoch {early_stopping.best_epoch}")
    
    # Test set evaluation
    print("\n🧪 Evaluating on TEST SET:")
    model.eval()
    test_correct = 0
    test_total = 0
    test_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predicted = (outputs.squeeze() >= CONFIG_OPTIMIZED['decision_threshold']).long()
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            test_probs.extend(outputs.squeeze().cpu().numpy().tolist())
    
    test_acc = 100.0 * test_correct / test_total
    
    print(f"  ✅ Test Accuracy: {test_acc:.2f}%")
    print(f"  ✅ Test Correct: {test_correct}/{test_total}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # SAVE TRAINING HISTORY & RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    results = {
        'config': CONFIG_OPTIMIZED,
        'best_epoch': early_stopping.best_epoch,
        'best_val_loss': float(early_stopping.best_loss),
        'test_accuracy': float(test_acc),
        'decision_threshold': float(CONFIG_OPTIMIZED['decision_threshold']),
        'history': {k: [float(v) if isinstance(v, (int, np.integer)) else v for v in vals] 
                   for k, vals in history.items()},
        'training_time': datetime.now().isoformat(),
    }
    
    results_dir = PROJECT_ROOT / 'outputs' / 'training_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'optimized_training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    plot_training_history(history, results_dir)
    
    print("\n" + "=" * 75)
    print("✅ TRAINING COMPLETE!")
    print("=" * 75)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Best Epoch: {early_stopping.best_epoch}")
    print(f"Best Val Loss: {early_stopping.best_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Model Saved: {CONFIG_OPTIMIZED['save_path']}")
    print("=" * 75 + "\n")
    
    return model, history, results


def plot_training_history(history, save_dir):
    """Create comprehensive training visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = history['epoch']
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['lr'], label='Learning Rate', marker='o', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio (validation/training)
    loss_ratio = [v/t if t > 0 else 1 for t, v in zip(history['train_loss'], history['val_loss'])]
    axes[1, 1].plot(epochs, loss_ratio, label='Val/Train Loss Ratio', marker='o', linewidth=2, color='red')
    axes[1, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Perfect Fit')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('Overfitting Monitor (Val/Train Loss)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / 'training_history_optimized.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training visualization saved: {plot_path}")
    plt.close()


if __name__ == '__main__':
    model, history, results = train_model_optimized()
