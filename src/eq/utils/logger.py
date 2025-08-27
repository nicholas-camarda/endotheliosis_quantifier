"""Logging utilities for the eq package."""

import logging
import sys
import time
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Set up logging for the eq package.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
        verbose: Enable verbose logging with more details
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        if verbose:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        else:
            format_string = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("eq")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create console handler with colors
    console_handler = ColoredStreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "eq") -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (default: "eq")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ColoredStreamHandler(logging.StreamHandler):
    """Custom stream handler with colored output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def emit(self, record):
        try:
            # Add color to the level name
            if record.levelname in self.COLORS:
                record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            
            super().emit(record)
        except Exception:
            self.handleError(record)


class ProgressLogger:
    """Logger with progress tracking capabilities."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, description: str = "Progress"):
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def step(self, message: str = None):
        """Log a progress step."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        
        if message:
            self.logger.info(f"[{self.current_step}/{self.total_steps}] ({progress:.1f}%) {message}")
        else:
            self.logger.info(f"[{self.current_step}/{self.total_steps}] ({progress:.1f}%) {self.description}")
    
    def complete(self, message: str = None):
        """Log completion."""
        elapsed = time.time() - self.start_time
        if message:
            self.logger.info(f"✅ Completed: {message} (took {elapsed:.2f}s)")
        else:
            self.logger.info(f"✅ {self.description} completed (took {elapsed:.2f}s)")


def log_function_call(func):
    """Decorator to log function calls with timing."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        logger.info(f"🚀 Starting {func_name}...")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"✅ {func_name} completed successfully (took {elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func_name} failed after {elapsed:.2f}s: {str(e)}")
            raise
    
    return wrapper


def log_data_info(data, name: str, logger: logging.Logger = None):
    """Log information about data structures."""
    if logger is None:
        logger = get_logger()
    
    if hasattr(data, 'shape'):
        logger.info(f"📊 {name} shape: {data.shape}")
    elif hasattr(data, '__len__'):
        logger.info(f"📊 {name} length: {len(data)}")
    else:
        logger.info(f"📊 {name} type: {type(data).__name__}")


def log_file_operation(operation: str, file_path: str, logger: logging.Logger = None):
    """Log file operations."""
    if logger is None:
        logger = get_logger()
    
    logger.info(f"📁 {operation}: {file_path}")


def log_model_info(model, name: str, logger: logging.Logger = None):
    """Log information about models."""
    if logger is None:
        logger = get_logger()
    
    if hasattr(model, 'summary'):
        logger.info(f"🤖 {name} model loaded successfully")
        # Log model summary if available
        try:
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            logger.debug(f"🤖 {name} model summary:\n" + "\n".join(summary_list))
        except:
            pass
    else:
        logger.info(f"🤖 {name} object loaded successfully")
