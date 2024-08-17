"""
Tests for `gabocutter` module.
"""

import pytest
import CryptoFraudDetection


def test_hello_world():
    assert CryptoFraudDetection.utils.misc.get_hello_world() == "Hello, World!"
