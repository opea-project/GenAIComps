import argparse
import asyncio
import signal
import time
from typing import TYPE_CHECKING, Optional, Union, Dict
from logger import Logger
from utils import is_port_free

if TYPE_CHECKING:  # pragma: no cover
    import threading
    import multiprocessing

HANDLED_SIGNALS = (
    signal.SIGINT,  # Unix signal 2. Sent by Ctrl+C.
    signal.SIGTERM,  # Unix signal 15. Sent by `kill <pid>`.
    signal.SIGSEGV,
)


class AsyncLoop:
    """
    Async loop to run a service asynchronously.
    """

    def __init__(self, 
                 args: Optional[Dict] = None,
                 cancel_event: Optional[
                     Union['asyncio.Event', 'multiprocessing.Event', 'threading.Event']
                     ] = None,
    ) -> None:
        self.args = args
        if args.get('name', None):
            self.name = f'{args.get("name")}/{self.__class__.__name__}'
        else:
            self.name = self.__class__.__name__
        self.protocol = args.get('protocol', 'http')
        self.host = args.get('host', 'localhost')
        self.port = args.get('port', 8080)
        self.quiet_error = args.get('quiet_error', False)
        self.logger = Logger(self.name)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.is_cancel = cancel_event or asyncio.Event()
        self.logger.info(f'Setting signal handlers')

        def _cancel(signum, frame):
                self.logger.info(f'Received signal {signum}')
                self.is_cancel.set(),

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _cancel)

        self._start_time = time.time()
        self._loop.run_until_complete(self.async_setup())

    def run_forever(self):
        """
        Running method to block the main thread.

        Run the event loop until a Future is done.
        """
        self._loop.run_until_complete(self._loop_body())

    def teardown(self):
        """Call async_teardown() and stop and close the event loop."""
        self._loop.run_until_complete(self.async_teardown())
        self._loop.stop()
        self._loop.close()
        self._stop_time = time.time()
        self.logger.info(f"Async loop is tore down. Duration: {self._stop_time - self._start_time}")

    def _get_server(self):
        # construct server type based on protocol (and potentially req handler class to keep backwards compatibility)
        if self.protocol.lower() == 'http':
            from http_service import HTTPService

            runtime_args = self.args.get('runtime_args', None)
            runtime_args['protocol'] = self.protocol
            runtime_args['host'] = self.host
            runtime_args['port'] = self.port
            return HTTPService(
                uvicorn_kwargs=self.args.get('uvicorn_kwargs', None),
                runtime_args=runtime_args,
                cors=self.args.get('cors', None),
            )

    async def async_setup(self):
        """
        The async method setup the runtime.

        Setup the uvicorn server.
        """
        if not (is_port_free(self.host, self.port)):
            raise RuntimeError(f'port:{self.port}')

        self.server = self._get_server()
        await self.server.setup_server()

    async def async_run_forever(self):
        """Running method of the server."""
        await self.server.run_server()

    async def async_teardown(self):
        """Shutdown the server."""
        await self.server.shutdown_server()

    async def _wait_for_cancel(self):
        """Do NOT override this method when inheriting from :class:`GatewayPod`"""
        # threads are not using asyncio.Event, but threading.Event
        if isinstance(self.is_cancel, asyncio.Event) and not hasattr(
            self.server, '_should_exit'
        ):
            await self.is_cancel.wait()
        else:
            while not self.is_cancel.is_set() and not getattr(
                self.server, '_should_exit', False
            ):
                await asyncio.sleep(0.1)

        await self.async_teardown()

    async def _loop_body(self):
        """Do NOT override this method when inheriting from :class:`GatewayPod`"""
        try:
            await asyncio.gather(self.async_run_forever(), self._wait_for_cancel())
        except asyncio.CancelledError:
            self.logger.warning('received terminate ctrl message from main process')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == KeyboardInterrupt:
            self.logger.info(f'{self!r} is interrupted by user')
        elif exc_type and issubclass(exc_type, Exception):
            self.logger.error(
                (
                    f'{exc_val!r} during {self.run_forever!r}'
                    + f'\n add "--quiet-error" to suppress the exception details'
                    if not self.quiet_error
                    else ''
                ),
                exc_info=not self.quiet_error,
            )
        else:
            self.logger.info(f'{self!r} is ended')
        try:
            # self.teardown()
            pass
        except OSError:
            # OSError(Stream is closed) already
            pass
        except Exception as ex:
            self.logger.error(
                (
                    f'{ex!r} during {self.teardown!r}'
                    + f'\n add "--quiet-error" to suppress the exception details'
                    if not self.args.quiet_error
                    else ''
                ),
                exc_info=not self.args.quiet_error,
            )

        return True
