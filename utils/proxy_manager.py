from dataclasses import dataclass, field
from typing import List, Optional
from anyio import Semaphore
import requests

from asyncio import Semaphore
from aiohttp import ClientSession


# We can implement NamedTuple too, but dataclasses is more faster
@dataclass(slots=True)
class ProxyDense:
    protocol: str
    login: str
    password: str
    ip_adress: str
    port: str
    session: Optional[ClientSession] = field(default=None)
    max_requests: Semaphore = field(default_factory=lambda: Semaphore(14))

    def __repr__(self) -> str:
        """
        Example: \\<protocol>://\\<login>:\\<password>@\\<ip_adress>:\\<port>
        """
        auth = self.login + ":" + self.password
        adress = self.ip_adress + ":" + self.port
        return self.protocol + "://" + auth + "@" + adress

    @classmethod
    def from_url(cls, url: str):
        protocol, url = url.split("://")
        auth, adress = url.split("@")
        login, password = auth.split(":")
        ip_adress, port = adress.split(":")

        return cls(protocol, login, password, ip_adress, port)

class ProxyManager:
    # TODO: Нужно сделать что-то вроде ProxyState, чтобы понимать какие прокси используются, а какие доступны.
    # Возможно стоит добавить таймауты на использование прокси
    def __init__(self) -> None:
        self.proxies: List[ProxyDense] = []
        self.__index: int = 0

    def _is_suitable(self, proxy: ProxyDense) -> bool | Exception:
        """Verify whether proxy can connect to the testing url"""
        try:
            resp = requests.get(
                url="https://httpbin.io/ip",
                proxies={proxy.protocol: str(proxy)}
            )
            assert proxy.ip_adress == resp.json()["origin"]
        except Exception as error:
            return error
        return True

    def add_proxy(
            self,
            url: Optional[str] = None,
            proxy: Optional[ProxyDense] = None,
            protocol: str = "",
            login: str = "", password: str = "",
            ip_adress: str = "", port: str = ""
    ) -> None:
        """Verify whether proxy can connect to the testing url and add it to list of proxies or raise exception"""
        if url:
            proxy = ProxyDense.from_url(url)
        elif not proxy:
            proxy = ProxyDense(protocol, login, password, ip_adress, port)

        test = self._is_suitable(proxy)

        if test == True:
            self.proxies.append(proxy)
        else:
            raise ConnectionError(
                f"Bad proxy {proxy.ip_adress}. Error: {test}"
            )

    def _switch_proxy(self) -> None:
        """Changes pointer"""
        if self.__index >= len(self.proxies):
            self.__index = 0
        else:
            self.__index += 1

    @property
    def proxy(self) -> Optional[str]:
        if self.proxies:
            proxy = str(self.proxies[self.__index])
            self._switch_proxy()
            return proxy
        return None
