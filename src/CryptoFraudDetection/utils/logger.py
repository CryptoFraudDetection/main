"""
File: logger.py

Description:
- This file contains the Logger class used for logging messages to the console and a log file.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class Logger:
    """
    Logger class to handle logging messages to both the console and a file.

    Attributes:
        logger (logging.Logger): The logger instance used for logging messages.

    Methods:
        info(message: str) -> None: Logs an informational message.
        warning(message: str) -> None: Logs a warning message.
        error(message: str) -> None: Logs an error message.
        critical(message: str) -> None: Logs a critical message.
    """

    def __init__(
        self, name: str, level: int = logging.DEBUG, log_dir: str = "logs"
    ) -> None:
        """
        Initializes the Logger instance with the specified name, log level, and log directory.

        Args:
            name (str): The name of the logger, typically the name of the module or class.
            level (int, optional): The log level (default: logging.DEBUG).
            log_dir (str, optional): The directory where log files will be saved (default: "logs").
        """
        self.logger: logging.Logger = logging.getLogger(name)

        if not isinstance(level, int):
            level = level.value
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set up console handler
        stream_handler: logging.StreamHandler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers only if there are no existing handlers (to avoid duplication)
        if not self.logger.hasHandlers():
            self.logger.addHandler(stream_handler)

            # Create logs directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Set up rotating file handler to handle log rotation and avoid large log files
            time: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_handler: RotatingFileHandler = RotatingFileHandler(
                f"{log_dir}/{time}_{name}.log", maxBytes=1024 * 1024 * 5, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str) -> None:
        """
        Logs an informational message.

        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """
        Logs a warning message.

        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """
        Logs an error message.

        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """
        Logs a critical message.

        Args:
            message (str): The message to log.
        """
        self.logger.critical(message)

    def handle_exception(
        self, exception_class: Exception, message: str, logger_level: str = "error"
    ) -> None:
        """
        Handles exception logging and raising.

        Args:
            exception_class (Exception): The class of the exception to raise.
            message (str): The error message to log and raise.
            logger_level (str): The logging level to use ("error", "warning", "info"). Defaults to "error".
        """
        if logger_level == "error":
            self.logger.error(message)
        elif logger_level == "warning":
            self.logger.warning(message)
        elif logger_level == "info":
            self.logger.info(message)

        raise exception_class(message)
