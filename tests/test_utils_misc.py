"""
This module contains the tests for the utils.misc module.
"""

from CryptoFraudDetection.utils.misc import get_hello_world


def test_hello_world():
    """
    Test the hello_world function from the utils.misc module.
    """
    assert get_hello_world() == "Hello, World!"
