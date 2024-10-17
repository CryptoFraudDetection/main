"""
File: enums.py

Description:
- This file contains the enums used in the project.
"""

from enum import Enum


class ScraperNotebookMode(Enum):
    """
    Enum for the notebook mode
    """

    WRITE = "write"
    READ = "read"


class LoggerMode(Enum):
    """
    Enum recreating for the logging package
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    FATAL = 50
