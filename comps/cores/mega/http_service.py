# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import FastAPI
from uvicorn import Config, Server

from .base_service import BaseService


class HTTPService(BaseService):
    """FastAPI HTTP service based on BaseService class.

    This property should return a fastapi app.
    """

    def __init__(
        self,
        uvicorn_kwargs: Optional[dict] = None,
        cors: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the HTTPService
        :param uvicorn_kwargs: Dictionary of kwargs arguments that will be passed to Uvicorn server when starting the server
        :param cors: If set, a CORS middleware is added to FastAPI frontend to allow cross-origin access.

        :param kwargs: keyword args
        """
        super().__init__(**kwargs)
        self.uvicorn_kwargs = uvicorn_kwargs or {}
        self.cors = cors
        self._app = self._create_app()

    @property
    def app(self):
        """Get the default base API app for Server
        :return: Return a FastAPI app for the default HTTPGateway."""
        return self._app

    def _create_app(self):
        """Create a FastAPI application.

        :return: a FastAPI application.
        """
        app = FastAPI(title=self.title, description=self.description)

        if self.cors:
            from fastapi.middleware.cors import CORSMiddleware

            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self.logger.info("CORS is enabled.")

        @app.get(
            path="/v1/health_check",
            summary="Get the status of GenAI microservice",
            tags=["Debug"],
        )
        async def _health_check():
            """Get the health status of this GenAI microservice."""
            return {"Service Title": self.title, "Service Description": self.description}

        return app

    async def initialize_server(self):
        """Initialize and return HTTP server."""
        self.logger.info("Setting up HTTP server")

        class UviServer(Server):
            """The uvicorn server."""

            async def setup_server(self, sockets=None):
                """Setup uvicorn server.

                :param sockets: sockets of server.
                """
                config = self.config
                if not config.loaded:
                    config.load()
                self.lifespan = config.lifespan_class(config)
                await self.startup(sockets=sockets)
                if self.should_exit:
                    return

            async def start_server(self, **kwargs):
                """Start the server.

                :param kwargs: keyword arguments
                """
                await self.main_loop()

        app = self.app

        self.server = UviServer(
            config=Config(
                app=app,
                host=self.host_address,
                port=self.primary_port,
                log_level="info",
                **self.uvicorn_kwargs,
            )
        )
        self.logger.info(f"Uvicorn server setup on port {self.primary_port}")
        await self.server.setup_server()
        self.logger.info("HTTP server setup successful")

    async def execute_server(self):
        """Run the HTTP server indefinitely."""
        await self.server.start_server()

    async def terminate_server(self):
        """Terminate the HTTP server and free resources allocated when setting up the server."""
        self.logger.info("Initiating server termination")
        self.server.should_exit = True
        await self.server.shutdown()
        self.logger.info("Server termination completed")

    @staticmethod
    def check_server_readiness(ctrl_address: str, timeout: float = 1.0, logger=None, **kwargs) -> bool:
        """Check if server status is ready.

        :param ctrl_address: the address where the control request needs to be sent
        :param timeout: timeout of the health check in seconds
        :param logger: Customized Logger to be used
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        import urllib.request
        from http import HTTPStatus

        try:
            conn = urllib.request.urlopen(url=f"http://{ctrl_address}", timeout=timeout)
            return conn.code == HTTPStatus.OK
        except Exception as exc:
            if logger:
                logger.info(f"Exception: {exc}")

            return False

    @staticmethod
    async def async_check_server_readiness(ctrl_address: str, timeout: float = 1.0, logger=None, **kwargs) -> bool:
        """Asynchronously check if server status is ready.

        :param ctrl_address: the address where the control request needs to be sent
        :param timeout: timeout of the health check in seconds
        :param logger: Customized Logger to be used
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        return HTTPService.check_server_readiness(ctrl_address, timeout, logger=logger)
