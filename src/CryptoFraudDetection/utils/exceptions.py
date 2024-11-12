"""
File: exceptions.py

Description:
- This file contains the exceptions used in the project.
"""


class DetectedBotException(Exception):
    """
    Exception raised when a bot is detected
    """

    def __init__(self, message="Detected as a bot"):
        self.message = message
        super().__init__(self.message)


class InvalidParameterException(Exception):
    """
    Exception raised when an invalid parameter is passed
    """

    def __init__(self, message="Invalid parameter"):
        self.message = message
        super().__init__(self.message)


class APIKeyNotSetException(Exception):
    """
    Exception raised when the API key is not set
    """

    def __init__(
        self,
        message="API Key not set, please set it in the .env file and load it using 'source .env'",
    ):
        self.message = message
        super().__init__(self.message)


class AuthenticationError(Exception):
    """
    Exception raised when authentication details are missing for scraping Twitter.
    """

    def __init__(
        self,
        message="Authentication details are required (either cookies or username/password).",
    ):
        self.message = message
        super().__init__(self.message)


class ProxyProtocolNotImplemented(Exception):
    """
    Exception raised when the proxy protocol is not implemented.
    """

    def __init__(self, message="Proxy protocol is not implemented."):
        self.message = message
        super().__init__(self.message)


class ProxyNotWorking(Exception):
    """
    Exception raised when the proxy is not working.
    """

    def __init__(self, message="Proxy is not working."):
        self.message = message
        super().__init__(self.message)


class ProxyIpEqualRealIp(Exception):
    """
    Exception raised when the proxy IP is equal to the real IP.
    """

    def __init__(self, message="Proxy IP is equal to the real IP."):
        self.message = message
        super().__init__(self.message)
