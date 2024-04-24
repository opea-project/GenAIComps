import abc
from logger import Logger
from types import SimpleNamespace
from typing import Dict, Optional, TYPE_CHECKING


__all__ = ['BaseServer']


class BaseService():
    """
    BaseService creates a HTTP/gRPC server as a microservice.
    """

    def __init__(
        self,
        name: Optional[str] = 'Base service',
        runtime_args: Optional[Dict] = None,
        **kwargs,
    ):
        self.name = name
        self.runtime_args = runtime_args
        self._reform_runtime_args()
        self.title = self.runtime_args.title
        self.description = self.runtime_args.description
        self.logger = Logger(self.name)
        self.server = None

    def _reform_runtime_args(self):
        _runtime_args = (
            self.runtime_args
            if isinstance(self.runtime_args, dict)
            else vars(self.runtime_args or {})
        )
        self.runtime_args = SimpleNamespace(**_runtime_args)

    @property
    def port(self):
        """Gets the first port of the port list argument. To be used in the regular case where a Gateway exposes a single port
        :return: The first port to be exposed
        """
        return (
            self.runtime_args.port[0]
            if isinstance(self.runtime_args.port, list)
            else self.runtime_args.port
        )

    @property
    def ports(self):
        """Gets all the list of ports from the runtime_args as a list.
        :return: The lists of ports to be exposed
        """
        return (
            self.runtime_args.port
            if isinstance(self.runtime_args.port, list)
            else [self.runtime_args.port]
        )

    @property
    def protocol(self):
        """Gets all the list of protocols from the runtime_args as a list.
        :return: The lists of protocols to be exposed
        """
        return (
            self.runtime_args.protocol
            if isinstance(self.runtime_args.protocol, list)
            else [self.runtime_args.protocol]
        )

    @property
    def host(self):
        """Gets the host from the runtime_args
        :return: The host where to bind the gateway
        """
        return self.runtime_args.host or '127.0.0.1'

    @abc.abstractmethod
    async def setup_server(self):
        """Setup server"""
        ...

    @abc.abstractmethod
    async def run_server(self):
        """Run server forever"""
        ...

    @abc.abstractmethod
    async def shutdown(self):
        """Shutdown the server and free other allocated resources, e.g, streamer object, health check service, ..."""
        ...

    @staticmethod
    def is_ready(
        ctrl_address: str,
        protocol: Optional[str] = 'http',
        **kwargs,
    ) -> bool:
        """
        Check if status is ready.
        :param ctrl_address: the address where the control request needs to be sent
        :param protocol: protocol of the gateway runtime
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        from http_service import HTTPService
        res = False
        if protocol is None or protocol == 'http':
            res = HTTPService.is_ready(ctrl_address)
        return res

    @staticmethod
    async def async_is_ready(
        ctrl_address: str,
        protocol: Optional[str] = 'grpc',
        **kwargs,
    ) -> bool:
        """
        Check if status is ready.
        :param ctrl_address: the address where the control request needs to be sent
        :param protocol: protocol of the gateway runtime
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        if TYPE_CHECKING:
            from http_service import HTTPService
        res = False
        if protocol is None or protocol == 'http':
            res = HTTPService.async_is_ready(ctrl_address)
        return res


