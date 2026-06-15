import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER LEARNING STRATEGY FOR YOUR PROJECT:
#
# Phase 1 (Epochs 1-5):   Freeze ALL ResNet layers, train only final layer
#                         WHY: Let new layer adapt without destroying
#                              pretrained features
#
# Phase 2 (Epochs 6-end): Unfreeze LAST block (layer4), train with tiny LR
#                         WHY: Fine-tune high-level features for fundus images
#                              while keeping low-level features (edges, etc.)
# ─────────────────────────────────────────────────────────────────────────────


def create_resnet50_model(num_classes=1, dropout_rate=0.5):
    """
    Create a modified ResNet-50 for glaucoma binary classification.

    CHANGES FROM ORIGINAL ResNet-50:
    1. Load pretrained ImageNet weights
    2. Add Dropout before final layer (prevents overfitting on small dataset)
    3. Replace final FC layer: 2048 → 512 → 1 (binary output)
    4. Use Sigmoid output (not Softmax) for binary classification

    WHY SIGMOID FOR BINARY?
    - Output is ONE number between 0 and 1
    - < 0.5 → Normal, >= 0.5 → Glaucoma
    - Works with Binary Cross-Entropy loss
    - Simpler and more standard than Softmax for 2 classes
    """

    print("Loading pretrained ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    print("[OK] Pretrained weights loaded!")

    # Freeze all layers initially.
    for param in model.parameters():
        param.requires_grad = False

    print("[LOCKED] All layers frozen")

    num_features = model.fc.in_features

    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid(),
    )

    print("[OK] New classification head added: 2048 -> 512 -> 1")
    print(f"   Dropout rates: {dropout_rate} (first), 0.3 (second)")

    return model


def freeze_all_except_head(model):
    """
    Phase 1: Freeze everything except the final FC layer.
    Use this for the first 5 epochs.
    """
    for name, param in model.named_parameters():
        if 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("[LOCKED] Phase 1: Frozen backbone")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def unfreeze_layer4_and_head(model):
    """
    Phase 2: Unfreeze layer4 (last residual block) + FC head.
    Use this after epoch 5 for fine-tuning.

    WHY ONLY LAYER4?
    - layer4 contains high-level features (shapes, structures)
    - These are most relevant for fundus image understanding
    - Earlier layers have low-level features (edges) — keep those frozen
    - Unfreezing everything would overfit on our small dataset
    """
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("[UNLOCKED] Phase 2: layer4 + FC unfrozen")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def print_model_summary(model):
    """Print a clean summary of the model layers and parameter counts."""
    print("\n" + "=" * 55)
    print("RESNET-50 MODEL SUMMARY")
    print("=" * 55)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    print("\nLayer groups:")

    layer_names = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
    for name in layer_names:
        layer = getattr(model, name, None)
        if layer:
            params = sum(p.numel() for p in layer.parameters())
            trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            status = "[UNLOCKED] TRAINABLE" if trainable > 0 else "[LOCKED] FROZEN"
            print(f"  {name:10s}: {params:>10,} params  {status}")
    print("=" * 55)


def build_model():
    """Backward-compatible helper that returns the configured model."""
    model = create_resnet50_model()
    freeze_all_except_head(model)
    return model


# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = create_resnet50_model()
    freeze_all_except_head(model)
    print_model_summary(model)

    # Test with a dummy image (batch=2, RGB, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("\nTest forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.detach().numpy().flatten()}")
    print("  (Values should be between 0 and 1)")
    print("\n[OK] Model created successfully!")
