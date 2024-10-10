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
