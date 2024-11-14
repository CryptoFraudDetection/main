"""
Miscellaneous utility functions for the scraper module.
"""

import json
import requests

from selenium import webdriver
from selenium.common.exceptions import WebDriverException, NoSuchElementException
from selenium.webdriver.common.proxy import Proxy, ProxyType

from CryptoFraudDetection.utils.exceptions import (
    ProxyIpEqualRealIp,
    ProxyProtocolNotImplemented,
    ProxyNotWorking,
)


def get_driver(
    headless: bool = False,
    proxy_protocol: str | None = None,
    proxy_address: str | None = None,
) -> webdriver.Firefox:
    """
    Return a Selenium Chrome WebDriver object.

    Args:
        headless (bool): Whether to run the browser in headless mode.
        proxy_protocol (str): The protocol of the proxy to use.
        proxy_address (str): The address of the proxy to use. <ip>:<port>

    Returns:
        WebDriver: A Selenium WebDriver object.
    """
    driver = None

    options = webdriver.FirefoxOptions()
    options.set_preference("devtools.jsonview.enabled", False)

    if headless:
        options.add_argument("--headless")

    if proxy_protocol and proxy_address:
        proxy = Proxy(
            {
                "proxyType": ProxyType.MANUAL,
            }
        )

        match proxy_protocol:
            case "http":
                proxy.http_proxy = proxy_address
                proxy.ssl_proxy = proxy_address
            case "socks4":
                proxy.socks_proxy = proxy_address
                proxy.socks_version = 4
            case "socks5":
                proxy.socks_proxy = proxy_address
                proxy.socks_version = 5
            case _:
                raise ProxyProtocolNotImplemented(
                    f"Proxy protocol {proxy_protocol} is not implemented."
                )

        options.proxy = proxy

        driver = webdriver.Firefox(options=options)
        try:
            driver.get("https://httpbin.io/ip")

            fetched_element = driver.find_element("tag name", "pre").text
            fetched_json = json.loads(fetched_element)
            fetched_ip = fetched_json["origin"].split(":")[0]

            real_ip = requests.get("https://api.ipify.org", timeout=10).text

            if fetched_ip == real_ip:
                raise ProxyIpEqualRealIp(
                    f"Proxy IP {fetched_ip} is equal to the real IP {real_ip}. The proxy is not working."
                )
        except (
            ProxyIpEqualRealIp,
            WebDriverException,
            NoSuchElementException,
        ) as e:
            driver.quit()
            raise ProxyNotWorking(
                f"Proxy {proxy_protocol}:{proxy_address} is not working."
            ) from e

    driver = webdriver.Firefox(options=options) if not driver else driver
    driver.set_page_load_timeout(60)

    return driver
