# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import multiprocessing
import os
import signal
from typing import Any, Optional, Type

from ..proto.docarray import TextDoc
from .constants import ServiceRoleType
from .utils import check_ports_availability

opea_microservices = {}


class MicroService:
    """MicroService class to create a microservice."""

    def __init__(
        self,
        name: Optional[str] = None,
        service_role: ServiceRoleType = ServiceRoleType.MICROSERVICE,
        protocol: str = "http",
        host: str = "localhost",
        port: int = 8080,
        expose_endpoint: Optional[str] = "/",
        input_datatype: Type[Any] = TextDoc,
        output_datatype: Type[Any] = TextDoc,
        replicas: int = 1,
        provider: Optional[str] = None,
        provider_endpoint: Optional[str] = None,
    ):
        """Init the microservice."""
        self.name = f"{name}/{self.__class__.__name__}" if name else self.__class__.__name__
        self.service_role = service_role
        self.protocol = protocol
        self.host = host
        self.port = port
        self.expose_endpoint = expose_endpoint
        self.input_datatype = input_datatype
        self.output_datatype = output_datatype
        self.replicas = replicas
        self.provider = provider
        self.provider_endpoint = provider_endpoint
        self.endpoints = []

        self.server = self._get_server()
        self.app = self.server.app
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_until_complete(self.async_setup())

    def _get_server(self):
        """Get the server instance based on the protocol.

        This method currently only supports HTTP services. It creates an instance of the HTTPService class with the
        necessary arguments.
        In the future, it will also support gRPC services.
        """
        from .http_service import HTTPService

        runtime_args = {
            "protocol": self.protocol,
            "host": self.host,
            "port": self.port,
            "title": self.name,
            "description": "OPEA Microservice Infrastructure",
        }

        return HTTPService(runtime_args=runtime_args)

    async def async_setup(self):
        """The async method setup the runtime.

        This method is responsible for setting up the server. It first checks if the port is available, then it gets
        the server instance and initializes it.
        """
        if self.protocol.lower() == "http":
            if not (check_ports_availability(self.host, self.port)):
                raise RuntimeError(f"port:{self.port}")

            await self.server.initialize_server()

    async def async_run_forever(self):
        """Running method of the server."""
        await self.server.execute_server()

    def run(self):
        """Running method to block the main thread.

        This method runs the event loop until a Future is done. It is designed to be called in the main thread to keep it busy.
        """
        self.event_loop.run_until_complete(self.async_run_forever())

    def start(self):
        self.process = multiprocessing.Process(target=self.run, daemon=False, name=self.name)
        self.process.start()

    async def async_teardown(self):
        """Shutdown the server."""
        await self.server.terminate_server()

    def stop(self):
        self.event_loop.run_until_complete(self.async_teardown())
        self.event_loop.stop()
        self.event_loop.close()
        self.server.logger.close()
        if self.process.is_alive():
            self.process.terminate()

    @property
    def endpoint_path(self):
        return f"{self.protocol}://{self.host}:{self.port}{self.expose_endpoint}"


def register_microservice(
    name: Optional[str] = None,
    service_role: ServiceRoleType = ServiceRoleType.MICROSERVICE,
    protocol: str = "http",
    host: str = "localhost",
    port: int = 8080,
    expose_endpoint: Optional[str] = "/",
    input_datatype: Type[Any] = TextDoc,
    output_datatype: Type[Any] = TextDoc,
    replicas: int = 1,
    provider: Optional[str] = None,
    provider_endpoint: Optional[str] = None,
):
    def decorator(func):
        micro_service = MicroService(
            name=name,
            service_role=service_role,
            protocol=protocol,
            host=host,
            port=port,
            expose_endpoint=expose_endpoint,
            input_datatype=input_datatype,
            output_datatype=output_datatype,
            replicas=replicas,
            provider=provider,
            provider_endpoint=provider_endpoint,
        )
        micro_service.app.router.add_api_route(expose_endpoint, func, methods=["POST"])
        opea_microservices[name] = micro_service
        return func

    return decorator
