from __future__ import annotations
from pathlib import Path
import os

# Project root (one level above src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Font asset (optional). Recommend bundling assets/fonts/times.ttf if you want
# Times New Roman consistently across platforms.
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
TIMES_TTF = FONTS_DIR / "times.ttf"

# Device helper: allow overriding with env var GLAUCOMA_DEVICE
def get_default_device():
    env = os.environ.get("GLAUCOMA_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs():
    for d in (OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
"""Centralized configuration management for glaucoma detection project."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing pipeline."""
    
    kernel_size: int = 5
    gaussian_sigma: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple[int, int] = (8, 8)
    target_value_range: tuple[float, float] = (0.0, 1.0)
    use_clahe: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number (3, 5, 7, ...)")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if self.clahe_clip_limit <= 0:
            raise ValueError("clahe_clip_limit must be positive")
        if len(self.clahe_tile_size) != 2 or any(x <= 0 for x in self.clahe_tile_size):
            raise ValueError("clahe_tile_size must be a tuple of 2 positive integers")


@dataclass
class SegmentationConfig:
    """Configuration for segmentation pipeline."""
    
    roi_size: tuple[int, int] = (200, 200)
    morph_kernel_size: tuple[int, int] = (5, 5)
    cup_threshold_percentile: float = 65.0
    min_disc_pixels: int = 100
    min_cup_area_ratio: float = 0.02
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if len(self.roi_size) != 2 or any(x <= 0 for x in self.roi_size):
            raise ValueError("roi_size must be a tuple of 2 positive integers")
        if self.cup_threshold_percentile < 0 or self.cup_threshold_percentile > 100:
            raise ValueError("cup_threshold_percentile must be between 0 and 100")
        if self.min_disc_pixels < 0:
            raise ValueError("min_disc_pixels must be non-negative")
        if self.min_cup_area_ratio < 0 or self.min_cup_area_ratio > 1:
            raise ValueError("min_cup_area_ratio must be between 0 and 1")


@dataclass
class CDRConfig:
    """Configuration for CDR calculation."""
    
    cdr_normal_threshold: float = 0.5
    cdr_borderline_threshold: float = 0.6
    cdr_glaucoma_threshold: float = 0.8
    
    def validate(self) -> None:
        """Validate CDR thresholds."""
        if not (0 <= self.cdr_normal_threshold <= self.cdr_borderline_threshold <= self.cdr_glaucoma_threshold <= 1.0):
            raise ValueError("CDR thresholds must be in order: normal < borderline < glaucoma")


@dataclass
class ModelConfig:
    """Configuration for ResNet-50 model."""
    
    num_classes: int = 1
    dropout_rate_1: float = 0.4
    dropout_rate_2: float = 0.3
    hidden_dim: int = 512
    use_pretrained: bool = True
    freeze_backbone: bool = True
    
    def validate(self) -> None:
        """Validate model configuration."""
        if self.num_classes < 1:
            raise ValueError("num_classes must be at least 1")
        if not (0 <= self.dropout_rate_1 < 1 and 0 <= self.dropout_rate_2 < 1):
            raise ValueError("dropout rates must be between 0 and 1")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    dataset_root: str | Path = PROJECT_ROOT / "dataset"
    batch_size: int = 16
    num_epochs: int = 50
    phase1_epochs: int = 8
    phase2_epochs: int = 42
    lr_phase1: float = 5e-4
    lr_phase2: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 12
    early_stop_min_delta: float = 5e-4
    focal_loss_alpha: float | None = None  # Auto-compute if None
    focal_loss_gamma: float = 2.0
    gradient_clip_max_norm: float = 1.0
    # Use helper at top-level to select device (resilient across Python versions)
    device: str = get_default_device()
    num_workers: int = 4 if device == "cuda" else 0
    seed: int = 42
    save_path: str | Path = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"
    resume_from: str | Path | None = None
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if not (0 < self.phase1_epochs <= self.num_epochs):
            raise ValueError("phase1_epochs must be <= num_epochs")
        if not (0 < self.phase2_epochs <= self.num_epochs):
            raise ValueError("phase2_epochs must be <= num_epochs")
        if self.phase1_epochs + self.phase2_epochs != self.num_epochs:
            raise ValueError("phase1_epochs + phase2_epochs must equal num_epochs")
        if self.lr_phase1 <= 0 or self.lr_phase2 <= 0:
            raise ValueError("Learning rates must be positive")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be at least 1")
        if self.focal_loss_gamma <= 0:
            raise ValueError("focal_loss_gamma must be positive")
        if self.gradient_clip_max_norm <= 0:
            raise ValueError("gradient_clip_max_norm must be positive")


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    supported_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    min_image_width: int = 200
    min_image_height: int = 200
    max_image_width: int = 4096
    max_image_height: int = 4096
    
    def validate(self) -> None:
        """Validate dataset configuration."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if self.min_image_width <= 0 or self.min_image_height <= 0:
            raise ValueError("Minimum image dimensions must be positive")
        if self.max_image_width <= self.min_image_width or self.max_image_height <= self.min_image_height:
            raise ValueError("Maximum image dimensions must be > minimum dimensions")


@dataclass
class OutputConfig:
    """Configuration for output directories and formats."""
    
    results_dir: str | Path = PROJECT_ROOT / "outputs" / "results"
    models_dir: str | Path = PROJECT_ROOT / "outputs" / "models"
    plots_dir: str | Path = PROJECT_ROOT / "outputs" / "plots"
    preprocessed_dir: str | Path = PROJECT_ROOT / "outputs" / "preprocessed"
    logs_dir: str | Path = PROJECT_ROOT / "outputs" / "logs"
    plot_dpi: int = 150
    plot_format: str = "png"
    save_intermediate: bool = False
    
    def validate(self) -> None:
        """Validate output configuration."""
        if self.plot_dpi <= 0:
            raise ValueError("plot_dpi must be positive")
        if self.plot_format not in ("png", "jpg", "pdf", "svg"):
            raise ValueError(f"plot_format must be one of: png, jpg, pdf, svg")


def _check_cuda_available() -> bool:
    """Check if CUDA is available (safely)."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self) -> None:
        """Initialize all configuration sections."""
        self.preprocessing = PreprocessingConfig()
        self.segmentation = SegmentationConfig()
        self.cdr = CDRConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.dataset = DatasetConfig()
        self.output = OutputConfig()
    
    def validate_all(self) -> None:
        """Validate all configuration sections."""
        print("Validating configuration...")
        self.preprocessing.validate()
        self.segmentation.validate()
        self.cdr.validate()
        self.model.validate()
        self.training.validate()
        self.dataset.validate()
        self.output.validate()
        print("✅ All configuration validated successfully!")
    
    def get_dict(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            "preprocessing": self.preprocessing,
            "segmentation": self.segmentation,
            "cdr": self.cdr,
            "model": self.model,
            "training": self.training,
            "dataset": self.dataset,
            "output": self.output,
        }
    
    def print_summary(self) -> None:
        """Print a summary of all configuration."""
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"\nPREPROCESSING:")
        print(f"  Kernel Size: {self.preprocessing.kernel_size}")
        print(f"  Gaussian Sigma: {self.preprocessing.gaussian_sigma}")
        print(f"  CLAHE Enabled: {self.preprocessing.use_clahe}")
        print(f"\nSEGMENTATION:")
        print(f"  ROI Size: {self.segmentation.roi_size}")
        print(f"  Cup Threshold Percentile: {self.segmentation.cup_threshold_percentile}")
        print(f"\nMODEL:")
        print(f"  Dropout Rates: {self.model.dropout_rate_1} / {self.model.dropout_rate_2}")
        print(f"  Pretrained: {self.model.use_pretrained}")
        print(f"\nTRAINING:")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Epochs: {self.training.num_epochs} (Phase1: {self.training.phase1_epochs}, Phase2: {self.training.phase2_epochs})")
        print(f"  Learning Rates: {self.training.lr_phase1} / {self.training.lr_phase2}")
        print(f"  Device: {self.training.device}")
        print(f"  Seed: {self.training.seed}")
        print(f"\nDATASET:")
        print(f"  Train/Val/Test: {self.dataset.train_split:.1%} / {self.dataset.val_split:.1%} / {self.dataset.test_split:.1%}")
        print("=" * 70)


# Global singleton instance
_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


if __name__ == "__main__":
    config = get_config()
    config.validate_all()
    config.print_summary()
