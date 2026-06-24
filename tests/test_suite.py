"""Comprehensive test suite for glaucoma detection project."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_config_initialization() -> None:
    """Test that configuration initializes without errors."""
    from config import get_config
    
    config = get_config()
    assert config is not None
    assert config.preprocessing is not None
    assert config.training is not None


def test_config_validation() -> None:
    """Test that configuration validation works."""
    from config import get_config
    
    config = get_config()
    try:
        config.validate_all()
    except Exception as e:
        pytest.fail(f"Configuration validation failed: {e}")


def test_preprocessing_config_invalid_kernel() -> None:
    """Test that invalid kernel size raises error."""
    from config import PreprocessingConfig
    
    with pytest.raises(ValueError):
        config = PreprocessingConfig(kernel_size=4)  # Even number
        config.validate()


def test_cdr_config_invalid_thresholds() -> None:
    """Test that invalid CDR thresholds raise error."""
    from config import CDRConfig
    
    with pytest.raises(ValueError):
        config = CDRConfig(
            cdr_normal_threshold=0.6,
            cdr_borderline_threshold=0.5,  # Out of order
            cdr_glaucoma_threshold=0.8,
        )
        config.validate()


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_logger_creation() -> None:
    """Test logger creation."""
    from error_handler import LoggerSetup
    
    logger = LoggerSetup.get_logger(__name__)
    assert logger is not None


def test_file_validation_missing_file() -> None:
    """Test that validation fails for missing files."""
    from error_handler import validate_file_exists
    
    result = validate_file_exists("/nonexistent/path/file.txt")
    assert result is False


def test_image_format_validation_valid() -> None:
    """Test image format validation for valid format."""
    from error_handler import validate_image_format
    
    result = validate_image_format("image.jpg")
    assert result is True


def test_image_format_validation_invalid() -> None:
    """Test image format validation for invalid format."""
    from error_handler import validate_image_format
    
    result = validate_image_format("image.txt", supported_formats=(".jpg", ".png"))
    assert result is False


def test_batch_processor_success() -> None:
    """Test batch processor with successful operations."""
    from error_handler import BatchProcessor
    
    processor = BatchProcessor()
    items = [1, 2, 3, 4, 5]
    
    results = processor.process_batch(
        items,
        lambda x: x * 2,
        continue_on_error=True,
    )
    
    assert len(results) == 5
    assert results == [2, 4, 6, 8, 10]
    assert processor.successful_count == 5
    assert len(processor.failed_items) == 0


def test_batch_processor_partial_failure() -> None:
    """Test batch processor with partial failures."""
    from error_handler import BatchProcessor
    
    processor = BatchProcessor()
    items = [1, 2, "invalid", 4, 5]
    
    def process(x: Any) -> int:
        return int(x) * 2
    
    results = processor.process_batch(
        items,
        process,
        continue_on_error=True,
    )
    
    assert processor.successful_count == 4
    assert len(processor.failed_items) == 1


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_image() -> None:
    """Test image normalization."""
    from preprocessing import normalize_image
    
    image = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = normalize_image(image)
    
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_gaussian_filter_invalid_kernel() -> None:
    """Test that invalid kernel size raises error."""
    from preprocessing import apply_gaussian_filter
    
    image = np.random.rand(100, 100)
    
    with pytest.raises(ValueError):
        apply_gaussian_filter(image, kernel_size=4)  # Even number


def test_green_channel_extraction() -> None:
    """Test green channel extraction."""
    from preprocessing import extract_green_channel
    
    # Create BGR image
    bgr_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    green = extract_green_channel(bgr_image)
    
    assert green.shape == (100, 100)
    np.testing.assert_array_equal(green, bgr_image[:, :, 1])


def test_green_channel_extraction_invalid_shape() -> None:
    """Test that invalid shape raises error."""
    from preprocessing import extract_green_channel
    
    with pytest.raises(ValueError):
        extract_green_channel(np.random.rand(100, 100))  # 2D instead of 3D


# ─────────────────────────────────────────────────────────────────────────────
# CDR CALCULATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_vertical_diameter_empty_mask() -> None:
    """Test vertical diameter with empty mask."""
    from cdr import compute_vertical_diameter
    
    mask = np.zeros((100, 100), dtype=np.uint8)
    diameter = compute_vertical_diameter(mask)
    
    assert diameter == 0


def test_vertical_diameter_full_mask() -> None:
    """Test vertical diameter with full mask."""
    from cdr import compute_vertical_diameter
    
    mask = np.ones((100, 100), dtype=np.uint8)
    diameter = compute_vertical_diameter(mask)
    
    assert diameter == 100


def test_cdr_computation_zero_disc() -> None:
    """Test CDR computation with zero disc diameter."""
    from cdr import compute_cdr
    
    disc_mask = np.zeros((100, 100), dtype=np.uint8)
    cup_mask = np.ones((50, 50), dtype=np.uint8)
    
    cdr, details = compute_cdr(disc_mask, cup_mask)
    
    assert cdr == 0.0
    assert details["disc_vertical_diameter"] == 0


def test_cdr_interpretation_normal() -> None:
    """Test CDR interpretation for normal case."""
    from cdr import interpret_cdr
    
    interpretation = interpret_cdr(0.4)
    
    assert interpretation["status"] == "NORMAL"
    assert interpretation["risk_level"] == "Low Risk"


def test_cdr_interpretation_glaucoma() -> None:
    """Test CDR interpretation for glaucoma case."""
    from cdr import interpret_cdr
    
    interpretation = interpret_cdr(0.7)
    
    assert interpretation["status"] == "GLAUCOMA SUSPECTED"
    assert interpretation["risk_level"] == "High Risk"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_model_creation() -> None:
    """Test ResNet-50 model creation."""
    import torch
    from model import create_resnet50_model
    
    model = create_resnet50_model()
    
    assert model is not None
    assert hasattr(model, 'fc')


def test_model_forward_pass() -> None:
    """Test model forward pass with dummy input."""
    import torch
    from model import create_resnet50_model
    
    model = create_resnet50_model()
    
    # Dummy input: batch_size=2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, 1)
    assert (output >= 0.0).all() and (output <= 1.0).all()


def test_model_freeze_head_only() -> None:
    """Test that freeze_all_except_head freezes correctly."""
    from model import create_resnet50_model, freeze_all_except_head
    
    model = create_resnet50_model()
    freeze_all_except_head(model)
    
    # Check that FC layer is trainable
    fc_trainable = any(p.requires_grad for name, p in model.named_parameters() if 'fc' in name)
    assert fc_trainable
    
    # Check that conv1 is frozen
    conv1_trainable = any(p.requires_grad for name, p in model.named_parameters() if 'conv1' in name)
    assert not conv1_trainable


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_with_synthetic_image() -> None:
    """Test full pipeline with synthetic image."""
    import tempfile
    from pathlib import Path
    from PIL import Image
    
    # Create temporary synthetic image
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy fundus-like image
        img = Image.new('RGB', (512, 512), color=(100, 100, 100))
        img_path = Path(tmpdir) / "test_image.jpg"
        img.save(img_path)
        
        # Test preprocessing
        from preprocessing import preprocess_image
        result = preprocess_image(str(img_path))
        
        assert 'pipeline_input' in result
        assert result['pipeline_input'].shape == (512, 512)


def test_empty_directory_handling() -> None:
    """Test handling of empty directories."""
    import tempfile
    from pathlib import Path
    
    from preprocessing import find_first_image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_first_image(Path(tmpdir))
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET UTILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_config_valid_splits() -> None:
    """Test that valid splits pass validation."""
    from config import DatasetConfig
    
    config = DatasetConfig(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )
    config.validate()  # Should not raise


def test_dataset_config_invalid_splits() -> None:
    """Test that invalid splits fail validation."""
    from config import DatasetConfig
    
    with pytest.raises(ValueError):
        config = DatasetConfig(
            train_split=0.6,
            val_split=0.2,
            test_split=0.3,  # Sum > 1.0
        )
        config.validate()


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_fundus_image_grayscale() -> None:
    """Test that grayscale/2D images are rejected safely."""
    from api import validate_fundus_image
    
    # Grayscale image (2D)
    img_gray = np.zeros((100, 100), dtype=np.uint8)
    is_valid, msg = validate_fundus_image(img_gray)
    assert not is_valid
    assert "grayscale" in msg.lower() or "channels" in msg.lower()


def test_validate_fundus_image_invalid_channels() -> None:
    """Test that images with wrong channel count are rejected."""
    from api import validate_fundus_image
    
    # 4 channels
    img_4ch = np.zeros((100, 100, 4), dtype=np.uint8)
    is_valid, msg = validate_fundus_image(img_4ch)
    assert not is_valid
    assert "channels" in msg.lower()


# ─────────────────────────────────────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

