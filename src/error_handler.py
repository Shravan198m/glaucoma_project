"""Error handling and logging utilities for glaucoma detection project."""

from __future__ import annotations

import logging
import logging.handlers
import sys
import traceback
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class GlaucomaProjectException(Exception):
    """Base exception for glaucoma project."""
    pass


class ImageProcessingError(GlaucomaProjectException):
    """Raised when image processing fails."""
    pass


class SegmentationError(GlaucomaProjectException):
    """Raised when segmentation fails."""
    pass


class ModelError(GlaucomaProjectException):
    """Raised when model loading/inference fails."""
    pass


class DatasetError(GlaucomaProjectException):
    """Raised when dataset handling fails."""
    pass


class ConfigurationError(GlaucomaProjectException):
    """Raised when configuration is invalid."""
    pass


class FileOperationError(GlaucomaProjectException):
    """Raised when file operations fail."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

class LoggerSetup:
    """Setup and manage logging for the project."""
    
    _loggers: dict[str, logging.Logger] = {}
    
    @staticmethod
    def get_logger(
        name: str,
        log_dir: Optional[Path | str] = None,
        level: int = logging.INFO,
    ) -> logging.Logger:
        """
        Get or create a logger.
        
        Args:
            name: Logger name (usually __name__)
            log_dir: Directory to save log files (None for console only)
            level: Logging level (default: INFO)
            
        Returns:
            Configured logger instance
        """
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler (always)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '[%(levelname)s][%(name)s][%(funcName)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"{name.replace('.', '_')}.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
            )
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s][%(name)s][%(funcName)s:%(lineno)d] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        LoggerSetup._loggers[name] = logger
        return logger


# ─────────────────────────────────────────────────────────────────────────────
# ERROR RECOVERY DECORATORS
# ─────────────────────────────────────────────────────────────────────────────

F = TypeVar('F', bound=Callable[..., Any])


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier (delay *= backoff after each retry)
        logger: Logger instance (optional)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            attempt = 0
            current_delay = delay
            
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if logger:
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                    
                    if attempt >= max_retries:
                        if logger:
                            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper  # type: ignore[return-value]
    return decorator


def handle_exceptions(
    exception_types: tuple[type[Exception], ...] | type[Exception] = Exception,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
    reraise: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to handle exceptions gracefully.
    
    Args:
        exception_types: Exception types to catch
        default_return: Value to return on exception
        logger: Logger instance (optional)
        reraise: Whether to re-raise the exception after logging
    """
    if not isinstance(exception_types, tuple):
        exception_types = (exception_types,)
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                error_msg = f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                if logger:
                    logger.error(error_msg)
                else:
                    print(f"ERROR: {error_msg}", file=sys.stderr)
                
                if reraise:
                    raise
                return default_return
        
        return wrapper  # type: ignore[return-value]
    return decorator


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str = "Input validation failed",
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator to validate function inputs before execution.
    
    Args:
        validation_func: Function to validate inputs
        error_message: Error message on validation failure
        logger: Logger instance (optional)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not validation_func(args, kwargs):
                msg = f"{error_message} in {func.__name__}"
                if logger:
                    logger.error(msg)
                raise ValueError(msg)
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore[return-value]
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def validate_file_exists(file_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> bool:
    """Check if file exists and log if not."""
    file_path = Path(file_path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        if logger:
            logger.error(msg)
        else:
            warnings.warn(msg)
        return False
    return True


def validate_directory_exists(
    dir_path: Union[str, Path],
    create: bool = False,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Check if directory exists and optionally create it."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        if create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if logger:
                    logger.info(f"Created directory: {dir_path}")
                return True
            except Exception as e:
                msg = f"Failed to create directory {dir_path}: {e}"
                if logger:
                    logger.error(msg)
                else:
                    warnings.warn(msg)
                return False
        else:
            msg = f"Directory not found: {dir_path}"
            if logger:
                logger.warning(msg)
            else:
                warnings.warn(msg)
            return False
    return True


def validate_image_format(
    file_path: Union[str, Path],
    supported_formats: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Validate if file has supported image format."""
    file_path = Path(file_path)
    if file_path.suffix.lower() not in supported_formats:
        msg = f"Unsupported image format: {file_path.suffix}. Supported: {supported_formats}"
        if logger:
            logger.error(msg)
        else:
            warnings.warn(msg)
        return False
    return True


def validate_numpy_array(
    array: Any,
    expected_dims: Optional[int] = None,
    expected_dtype: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Validate numpy array properties."""
    try:
        import numpy as np
        if not isinstance(array, np.ndarray):
            raise TypeError("Input is not a numpy array")
        
        if expected_dims is not None and array.ndim != expected_dims:
            raise ValueError(f"Expected {expected_dims}D array, got {array.ndim}D")
        
        if expected_dtype is not None and array.dtype != expected_dtype:
            raise TypeError(f"Expected dtype {expected_dtype}, got {array.dtype}")
        
        return True
    except Exception as e:
        msg = f"Array validation failed: {e}"
        if logger:
            logger.error(msg)
        else:
            warnings.warn(msg)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# BATCH OPERATION SAFETY
# ─────────────────────────────────────────────────────────────────────────────

class BatchProcessor:
    """Safe batch processing with error tracking."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize batch processor."""
        self.logger = logger
        self.failed_items: list[dict[str, Any]] = []
        self.successful_count: int = 0
    
    def process_batch(
        self,
        items: list[Any],
        process_func: Callable[[Any], Any],
        continue_on_error: bool = True,
    ) -> list[Any]:
        """
        Process batch of items with error handling.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            continue_on_error: Continue processing if one item fails
            
        Returns:
            List of successfully processed items
        """
        results = []
        self.failed_items = []
        self.successful_count = 0
        
        for idx, item in enumerate(items):
            try:
                result = process_func(item)
                results.append(result)
                self.successful_count += 1
            except Exception as e:
                error_info = {
                    "index": idx,
                    "item": item,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                self.failed_items.append(error_info)
                
                if self.logger:
                    self.logger.error(
                        f"Failed to process item {idx}: {e}"
                    )
                
                if not continue_on_error:
                    raise
        
        if self.logger:
            self.logger.info(
                f"Batch processing complete: {self.successful_count} succeeded, "
                f"{len(self.failed_items)} failed"
            )
        
        return results
    
    def get_summary(self) -> dict[str, Any]:
        """Get processing summary."""
        return {
            "successful": self.successful_count,
            "failed": len(self.failed_items),
            "total": self.successful_count + len(self.failed_items),
            "failed_items": self.failed_items,
        }


if __name__ == "__main__":
    # Test logging
    logger = LoggerSetup.get_logger(__name__)
    logger.info("Logger initialized successfully!")
