"""
Logging configuration for PDPTWT solver.

This module sets up structured logging using loguru.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    console: bool = True,
    format_style: str = "detailed"
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = no file logging)
        console: Whether to log to console
        format_style: Format style (simple, detailed, json)
    """
    # Remove default handler
    logger.remove()
    
    # Define formats
    formats = {
        "simple": "<level>{level: <8}</level> | <level>{message}</level>",
        "detailed": (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        "json": "{time} {level} {name} {function} {line} {message}"
    }
    
    format_str = formats.get(format_style, formats["detailed"])
    
    # Add console handler
    if console:
        logger.add(
            sys.stderr,
            format=format_str,
            level=level,
            colorize=True
        )
    
    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=format_str,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
        
        logger.info(f"Logging to file: {log_path}")


def get_logger(name: str):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
