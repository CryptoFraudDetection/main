"""
File: exceptions.py

Description:
- This file contains the exceptions used in the project.
"""


class DetectedBotException(Exception):
    def __init__(self, message="Detected as a bot"):
        self.message = message
        super().__init__(self.message)


class InvalidParameterException(Exception):
    def __init__(self, message="Invalid parameter"):
        self.message = message
        super().__init__(self.message)
