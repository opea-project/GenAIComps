import os
from typing import Optional
from base_service import BaseService
from fastapi import FastAPI


class HTTPService(BaseService):
    """FastAPI HTTP service based on BaseService class. This property should return a fastapi app."""

    def __init__(
        self,
        uvicorn_kwargs: Optional[dict] = None,
        cors: Optional[bool] = False,
        **kwargs,
    ):
        """Initialize the FastAPIBaseGateway
        :param uvicorn_kwargs: Dictionary of kwargs arguments that will be passed to Uvicorn server when starting the server
        :param title: The title of this HTTP server. It will be used in automatics docs such as Swagger UI.
        :param description: The description of this HTTP server. It will be used in automatics docs such as Swagger UI.
        :param expose_endpoints: A JSON string that represents a map from executor endpoints (`@requests(on=...)`) to HTTP endpoints.
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
        :return: Return a FastAPI app for the default HTTPGateway
        """
        return self._app

    async def setup_server(self):
        """
        Initialize and return GRPC server
        """
        self.logger.info(f'Setting up HTTP server')
        from uvicorn import Config, Server

        class UviServer(Server):
            """The uvicorn server."""

            async def setup(self, sockets=None):
                """
                Setup uvicorn server.

                :param sockets: sockets of server.
                """
                config = self.config
                if not config.loaded:
                    config.load()
                self.lifespan = config.lifespan_class(config)
                await self.startup(sockets=sockets)
                if self.should_exit:
                    return

            async def serve(self, **kwargs):
                """
                Start the server.

                :param kwargs: keyword arguments
                """
                await self.main_loop()

        # app property will generate a new fastapi app each time called
        app = self.app
        
        self.server = UviServer(
            config=Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level='info',
                **self.uvicorn_kwargs,
            )
        )
        self.logger.info(f'UviServer server setup on port {self.port}')
        await self.server.setup()
        self.logger.info(f'HTTP server setup successful')

    async def shutdown_server(self):
        """
        Free resources allocated when setting up HTTP server
        """
        self.logger.info(f'Shutting down server')
        await super().shutdown()
        self.server.should_exit = True
        await self.server.shutdown()
        self.logger.info(f'Server shutdown finished')

    async def run_server(self):
        """Run HTTP server forever"""
        await self.server.serve()

    def _create_app(self):
        from fastapi.middleware.cors import CORSMiddleware
        _version = '0.0.1'

        app = FastAPI(
            title=self.title or 'My GenAI Micro Service',
            description=self.description
            or 'This is my awesome service. You can set `title` and `description` '
            'to customize the title and description.',
            version=_version,
        )

        if self.cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=['*'],
                allow_credentials=True,
                allow_methods=['*'],
                allow_headers=['*'],
            )
            self.logger.warning('CORS is enabled. This service is accessible from any website!')

        @app.get(
            path='/v1/health_check',
            summary='Get the status of GenAI microservice',
            tags=['Debug'],
        )
        async def _health_check():
            """
            Get the health status of this GenAI microservice.
            """
            return {
                'Service Title': self.title,
                'Service Description': self.description,
                'Service Version': _version
            }
        
        return app
    
    def _add_route(self, 
                   endpoint_path, 
                   route_method,
                   input_type):
        app_kwargs = dict(
            path=f'/{endpoint_path.strip("/")}',
            methods=[route_method],
            summary=f'Endpoint {endpoint_path}'
        )

        app = self.app
        @app.api_route(**app_kwargs)
        async def new_route(input: input_type):
            pass

    @staticmethod
    def is_ready(
        ctrl_address: str, timeout: float = 1.0, logger=None, **kwargs
    ) -> bool:
        """
        Check if status is ready.
        :param ctrl_address: the address where the control request needs to be sent
        :param timeout: timeout of the health check in seconds
        :param logger: JinaLogger to be used
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        import urllib.request
        from http import HTTPStatus

        try:
            conn = urllib.request.urlopen(url=f'http://{ctrl_address}', timeout=timeout)
            return conn.code == HTTPStatus.OK
        except Exception as exc:
            if logger:
                logger.info(f'Exception: {exc}')

            return False

    @staticmethod
    async def async_is_ready(
        ctrl_address: str, timeout: float = 1.0, logger=None, **kwargs
    ) -> bool:
        """
        Async Check if status is ready.
        :param ctrl_address: the address where the control request needs to be sent
        :param timeout: timeout of the health check in seconds
        :param logger: JinaLogger to be used
        :param kwargs: extra keyword arguments
        :return: True if status is ready else False.
        """
        return HTTPService.is_ready(ctrl_address, timeout, logger=logger)

