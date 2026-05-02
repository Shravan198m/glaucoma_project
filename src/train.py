import copy
import os
import sys
import time
from pathlib import Path

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

from dataset import create_dataloaders
from model import (
    create_resnet50_model,
    freeze_all_except_head,
    print_model_summary,
    unfreeze_layer4_and_head,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION — Tuned for REFUGE + CPU
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    'dataset_root': str(PROJECT_ROOT / 'dataset'),
    'batch_size': 8,              # Small = safe for CPU RAM
    'num_epochs': 20,             # 20 epochs total
    'phase1_epochs': 5,           # Epochs 1-5: frozen backbone
    'phase2_epochs': 15,          # Epochs 6-20: fine-tuning layer4
    'lr_phase1': 1e-3,            # Higher LR for new head (phase 1)
    'lr_phase2': 1e-4,            # Lower LR for fine-tuning (phase 2)
    'dropout': 0.5,
    'early_stop_patience': 6,     # Stop if no improvement for 6 epochs
    'save_path': str(PROJECT_ROOT / 'outputs' / 'models' / 'best_model.pth'),
    'device': 'cpu',              # CPU training
}


# ── LOSS FUNCTION ─────────────────────────────────────────────────────────────
def get_loss_function(class_weights):
    """
    Binary Cross-Entropy Loss with class-aware sample weighting.

    The current model head ends with Sigmoid, so binary cross-entropy is the
    correct companion loss here. We apply the dataset class weights per sample
    so the minority class has a stronger contribution to the objective.
    """
    class_weights = class_weights.float()
    normal_weight = class_weights[0].item()
    glaucoma_weight = class_weights[1].item()
    print(f"  Loss weight (normal):   {normal_weight:.4f}")
    print(f"  Loss weight (glaucoma): {glaucoma_weight:.4f}")

    def criterion(outputs, labels):
        sample_weights = torch.where(
            labels >= 0.5,
            torch.as_tensor(glaucoma_weight, device=outputs.device, dtype=outputs.dtype),
            torch.as_tensor(normal_weight, device=outputs.device, dtype=outputs.dtype),
        )
        return F.binary_cross_entropy(outputs, labels, weight=sample_weights)

    return criterion


# ── TRAINING ONE EPOCH ───────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Run one complete training epoch."""
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
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Batch [{batch_idx + 1}/{batch_count}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / batch_count
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ── VALIDATION ONE EPOCH ─────────────────────────────────────────────────────
def validate_one_epoch(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


# ── EARLY STOPPING ────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=6, min_delta=0.001, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model = None
        self.stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(self.best_model, self.save_path)
            print(f"    💾 Best model saved! (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            print(f"    ⏳ No improvement. Patience: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.stop = True
                print("    🛑 Early stopping triggered!")


# ── MAIN TRAINING FUNCTION ────────────────────────────────────────────────────
def train_model():
    """Complete training pipeline with two phases."""
    print("\n" + "=" * 60)
    print("GLAUCOMA DETECTION — TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {CONFIG['device'].upper()}")
    print(f"Total epochs: {CONFIG['num_epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print("=" * 60)

    device = torch.device(CONFIG['device'])

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        dataset_root=CONFIG['dataset_root'],
        batch_size=CONFIG['batch_size'],
    )

    model = create_resnet50_model(dropout_rate=CONFIG['dropout'])
    model = model.to(device)

    criterion = get_loss_function(class_weights)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': [],
    }

    early_stopping = EarlyStopping(
        patience=CONFIG['early_stop_patience'],
        save_path=CONFIG['save_path'],
    )

    print("\n" + "─" * 60)
    print("PHASE 1: Training classification head only (Epochs 1-5)")
    print("─" * 60)

    freeze_all_except_head(model)
    print_model_summary(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_phase1'],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
    )

    for epoch in range(1, CONFIG['phase1_epochs'] + 1):
        start_time = time.time()
        print(f"\n[PHASE 1] Epoch {epoch}/{CONFIG['phase1_epochs']}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print("\n  ┌─────────────────────────────────────────┐")
        print(f"  │ Epoch {epoch:2d} Summary                        │")
        print(f"  │ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%  │")
        print(f"  │ Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%  │")
        print(f"  │ Time: {elapsed:.1f}s                              │")
        print("  └─────────────────────────────────────────┘")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.stop:
            print("Early stopping in Phase 1.")
            break

    print("\n" + "─" * 60)
    print("PHASE 2: Fine-tuning layer4 + head (Epochs 6-20)")
    print("─" * 60)

    unfreeze_layer4_and_head(model)
    print_model_summary(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr_phase2'],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    early_stopping.counter = 0
    early_stopping.stop = False

    for epoch in range(CONFIG['phase1_epochs'] + 1, CONFIG['num_epochs'] + 1):
        start_time = time.time()
        phase2_ep = epoch - CONFIG['phase1_epochs']
        print(
            f"\n[PHASE 2] Epoch {epoch}/{CONFIG['num_epochs']} "
            f"(Phase2 epoch {phase2_ep}/{CONFIG['phase2_epochs']})"
        )
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start_time

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print("\n  ┌─────────────────────────────────────────┐")
        print(f"  │ Epoch {epoch:2d} Summary                        │")
        print(f"  │ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%  │")
        print(f"  │ Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%  │")
        print(f"  │ Time: {elapsed:.1f}s                              │")
        print("  └─────────────────────────────────────────┘")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.stop:
            print("Early stopping triggered. Training complete.")
            break

    plot_training_history(history)

    print("\n✅ Training complete!")
    print(f"Best model saved at: {CONFIG['save_path']}")

    return model, history, test_loader, class_weights


def plot_training_history(history):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2, markersize=4)
    axes[0].axvline(x=5.5, color='gray', linestyle='--', label='Phase 1→2 boundary')
    axes[0].set_title('Loss vs Epoch', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', linewidth=2, markersize=4)
    axes[1].axvline(x=5.5, color='gray', linestyle='--', label='Phase 1→2 boundary')
    axes[1].axhline(y=88, color='green', linestyle=':', label='Target (88%)')
    axes[1].set_title('Accuracy vs Epoch', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=15, fontweight='bold')
    plt.tight_layout()

    os.makedirs('outputs/plots', exist_ok=True)
    plt.savefig('outputs/plots/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('📊 Training history saved to outputs/plots/training_history.png')


def main():
    smoke_test = os.environ.get('GLAUCOMA_TRAIN_SMOKE_TEST', '').strip().lower() in {'1', 'true', 'yes'}

    if smoke_test:
        print('Running smoke test mode for quick validation.')
        print(f"Device: {CONFIG['device'].upper()}")
        print(f"Batch size: {CONFIG['batch_size']}")
        print(f"Planned epochs: {CONFIG['num_epochs']}")
        print('Smoke test mode does not start the full CPU training run.')
        return

    train_model()


if __name__ == '__main__':
    main()
