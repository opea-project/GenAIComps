# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
from collections import defaultdict, deque
from enum import Enum
from typing import Any, List, Optional, Type

from ..proto.docarray import TextDoc
from .constants import ServiceRoleType, ServiceType
from .utils import check_ports_availability

opea_microservices = {}


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
        static_batching: bool = False,
        static_batching_timeout: int = 1,
        static_batching_max_batch_size: int = 32,
    ):
        """Init the microservice."""
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
        self.static_batching = static_batching
        self.static_batching_timeout = static_batching_timeout
        self.static_batching_max_batch_size = static_batching_max_batch_size
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
            # create a batch request processor loop if using static batching
            if self.static_batching:
                self.buffer_lock = asyncio.Lock()
                self.request_buffer = defaultdict(deque)

                @self.app.on_event("startup")
                async def startup_event():
                    asyncio.create_task(self._static_batch_processor())

            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_until_complete(self._async_setup())

    async def _static_batch_processor(self):
        while True:
            await asyncio.sleep(self.static_batching_timeout)
            runtime_batch = {}  # {ServiceType.Embedding: [{"request": xx, "response": yy}, {}]}

            async with self.buffer_lock:
                # prepare the runtime batch, access to buffer is locked
                if self.request_buffer:
                    # grab max(MAX_BATCH_SIZE, REQUEST_SIZE) requests from buffer

                    for service_type, request_lst in self.request_buffer.items():
                        batch = []
                        for _ in range(max(self.static_batching_max_batch_size, len(request_lst))):
                            batch.append(request_lst.popleft())

                        runtime_batch[service_type] = batch

            # Run batched inference on the batch and set results
            for service_type, batch in runtime_batch:
                if not batch:
                    continue
                results = await self.static_batching_infer(service_type, batch)

                for req, result in zip(batch, results):
                    req["response"].set_result(result)

    async def static_batching_infer(self, service_type: Enum, batch: list[dict]):
        """Need to implement."""
        raise NotImplementedError("Unimplemented static batching inference!")

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
    static_batching: bool = False,
    static_batching_timeout: int = 1,
    static_batching_max_batch_size: int = 32,
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
                static_batching=static_batching,
                static_batching_timeout=static_batching_timeout,
                static_batching_max_batch_size=static_batching_max_batch_size,
            )
            opea_microservices[name] = micro_service
        opea_microservices[name].app.router.add_api_route(endpoint, func, methods=methods)

        return func

    return decorator
