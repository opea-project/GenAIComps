# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import os
from functools import wraps

from opentelemetry import trace
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ..mega.logger import CustomLogger

logger = CustomLogger("OpeaComponent")


def detach_ignore_err(self, token: object) -> None:
    """Resets Context to a previous value.

    Args:
        token: A reference to a previous Context.
    """
    try:
        self._current_context.reset(token)  # type: ignore
    except Exception as e:
        pass


# bypass the ValueError that ContextVar context was created in a different Context from StreamingResponse
ContextVarsRuntimeContext.detach = detach_ignore_err

resource = Resource.create({SERVICE_NAME: "opea"})
traceProvider = TracerProvider(resource=resource)

ENABLE_OPEA_TELEMETRY = False
telemetry_endpoint = os.environ.get("TELEMETRY_ENDPOINT")
if telemetry_endpoint is not None:

    ENABLE_OPEA_TELEMETRY = True
    logger.info(f" Has Telemetry Endpoint :  {telemetry_endpoint}")
    traceProvider.add_span_processor(BatchSpanProcessor(HTTPSpanExporter(endpoint=telemetry_endpoint)))

in_memory_exporter = InMemorySpanExporter()
traceProvider.add_span_processor(BatchSpanProcessor(in_memory_exporter))
trace.set_tracer_provider(traceProvider)

tracer = trace.get_tracer(__name__)


def opea_telemetry(func):
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(func.__qualname__) if ENABLE_OPEA_TELEMETRY else contextlib.nullcontext():
                res = await func(*args, **kwargs)
            return res

    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(func.__qualname__) if ENABLE_OPEA_TELEMETRY else contextlib.nullcontext():
                res = func(*args, **kwargs)
            return res

    return wrapper
