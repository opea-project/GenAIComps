# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
import requests
import time
from typing import Any, List, Optional, Type

from ..proto.docarray import TextDoc
from .constants import ServiceRoleType, ServiceType
from .utils import check_ports_availability
from .logger import CustomLogger

opea_microservices = {}

logger = CustomLogger("micro-service")
class MicroService:
    """MicroService class to create a microservice."""

    def __init__(
        self,
        name: Optional[str] = None,
        service_role: ServiceRoleType = ServiceRoleType.MICROSERVICE,
        service_type: ServiceType = ServiceType.LLM,
        protocol: str = "http",
        host: str = "localhost",
        port: int = 8080,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        endpoint: Optional[str] = "/",
        input_datatype: Type[Any] = TextDoc,
        output_datatype: Type[Any] = TextDoc,
        provider: Optional[str] = None,
        provider_endpoint: Optional[str] = None,
        use_remote_service: Optional[bool] = False,
        methods: List[str] = ["POST"],
        llm_endpoint: Optional[str] = None,
    ):
        """Init the microservice."""

        # Make sure LLM/Microservice endpoint are ready
        logger.info(f"Init microservice - {name}")
        if llm_endpoint is not None:
            logger.info(f"LLM Endpoint requested for microservice @ {llm_endpoint}")
            success = False
            while success is False:
                try:
                    for item in methods:
                        # Do we want to cater for GET request?
                        # if yes, how to make it clean
                        if item.upper() == "POST":
                            response = requests.post(llm_endpoint)
                            if response.status_code == 200:
                                logger.info(f"LLM endpoint is ready - {llm_endpoint}")
                            else:
                                logger.info(f"LLM endpoint is ready - but error status code - {llm_endpoint}")
                            success = True
                        
                except requests.exceptions.RequestException as e:
                    logger.info(f"Error: {e} - {llm_endpoint}")
                    time.sleep(2.5)
        else:
            url = f"{host}:{port}{endpoint}"
            if protocol.upper() == "HTTP":
                url = f"http://{url}"
            elif protocol.upper() == "HTTPS":
                url = f"https://{url}"
            # enable future support for gRPC?

            logger.info(f"Endpoint requested for microservice @ {url}")
            success = False
            while success is False:
                try:
                    for item in methods:
                        # Do we want to cater for GET request?
                        # if yes, how to make it clean
                        if item.upper() == "POST":
                            response = requests.post(url)
                            if response.status_code == 200:
                                logger.info(f"Microservice endpoint is ready - {url}")
                            else:
                                logger.info(f"Microservice endpoint is ready - but error status code - {url}")
                            success = True
                        else:
                            logger.info(f"Microservice endpoint method not supported - Skipping - {url}")
                            success = True
                except requests.exceptions.RequestException as e:
                    logger.info(f"Error: {e} - {url}")
                    time.sleep(2.5)

        self.name = f"{name}/{self.__class__.__name__}" if name else self.__class__.__name__
        self.service_role = service_role
        self.service_type = service_type
        self.protocol = protocol
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.input_datatype = input_datatype
        self.output_datatype = output_datatype
        self.use_remote_service = use_remote_service
        self.uvicorn_kwargs = {}

        if ssl_keyfile:
            self.uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

        if ssl_certfile:
            self.uvicorn_kwargs["ssl_certfile"] = ssl_certfile

        if not use_remote_service:
            self.provider = provider
            self.provider_endpoint = provider_endpoint
            self.endpoints = []

            self.server = self._get_server()
            self.app = self.server.app
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(self._async_setup())

    def _validate_env(self):
        """Check whether to use the microservice locally."""
        if self.use_remote_service:
            raise Exception(
                "Method not allowed for a remote service, please "
                "set use_remote_service to False if you want to use a local micro service!"
            )

    def _get_server(self):
        """Get the server instance based on the protocol.

        This method currently only supports HTTP services. It creates an instance of the HTTPService class with the
        necessary arguments.
        In the future, it will also support gRPC services.
        """
        self._validate_env()
        from .http_service import HTTPService

        runtime_args = {
            "protocol": self.protocol,
            "host": self.host,
            "port": self.port,
            "title": self.name,
            "description": "OPEA Microservice Infrastructure",
        }

        return HTTPService(uvicorn_kwargs=self.uvicorn_kwargs, runtime_args=runtime_args)

    async def _async_setup(self):
        """The async method setup the runtime.

        This method is responsible for setting up the server. It first checks if the port is available, then it gets
        the server instance and initializes it.
        """
        self._validate_env()
        if self.protocol.lower() == "http":
            if not (check_ports_availability(self.host, self.port)):
                raise RuntimeError(f"port:{self.port}")

            await self.server.initialize_server()

    async def _async_run_forever(self):
        """Running method of the server."""
        self._validate_env()
        await self.server.execute_server()

    def run(self):
        """Running method to block the main thread.

        This method runs the event loop until a Future is done. It is designed to be called in the main thread to keep it busy.
        """
        self._validate_env()
        self.event_loop.run_until_complete(self._async_run_forever())

    def start(self):
        self._validate_env()
        self.process = multiprocessing.Process(target=self.run, daemon=False, name=self.name)
        self.process.start()

    async def _async_teardown(self):
        """Shutdown the server."""
        self._validate_env()
        await self.server.terminate_server()

    def stop(self):
        self._validate_env()
        self.event_loop.run_until_complete(self._async_teardown())
        self.event_loop.stop()
        self.event_loop.close()
        self.server.logger.close()
        if self.process.is_alive():
            self.process.terminate()

    @property
    def endpoint_path(self):
        return f"{self.protocol}://{self.host}:{self.port}{self.endpoint}"


def register_microservice(
    name: Optional[str] = None,
    service_role: ServiceRoleType = ServiceRoleType.MICROSERVICE,
    service_type: ServiceType = ServiceType.UNDEFINED,
    protocol: str = "http",
    host: str = "localhost",
    port: int = 8080,
    ssl_keyfile: Optional[str] = None,
    ssl_certfile: Optional[str] = None,
    endpoint: Optional[str] = "/",
    input_datatype: Type[Any] = TextDoc,
    output_datatype: Type[Any] = TextDoc,
    provider: Optional[str] = None,
    provider_endpoint: Optional[str] = None,
    methods: List[str] = ["POST"],
    llm_endpoint: Optional[str] = None,
):
    def decorator(func):
        if name not in opea_microservices:
            micro_service = MicroService(
                name=name,
                service_role=service_role,
                service_type=service_type,
                protocol=protocol,
                host=host,
                port=port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                endpoint=endpoint,
                input_datatype=input_datatype,
                output_datatype=output_datatype,
                provider=provider,
                provider_endpoint=provider_endpoint,
                methods=methods,
                llm_endpoint=llm_endpoint,
            )
            opea_microservices[name] = micro_service
        opea_microservices[name].app.router.add_api_route(endpoint, func, methods=methods)

        return func

    return decorator
